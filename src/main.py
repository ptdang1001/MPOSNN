# -*-coding:utf-8-*-


# build-in libraries
import sys
import os
import argparse
# import time
import shutil


# Third-party libraries
import numpy as np
# import pandas as pd



# my libraries
#from utils.data_interface import load_solutions
from utils.data_interface import load_module
from utils.data_interface import load_geneExpression
from utils.data_interface import load_modulesGenes
from utils.data_interface import save_samples_modules
from utils.data_interface import get_std_scale_imbalanceLoss_realData
from utils.data_interface import plot_std_scale_imbalance_in_one
from utils.data_interface import get_imbalanceLoss
# from utils.data_interface import add_noise_to_geneExpression
#from utils.data_interface import generate_noise
from utils.data_interface import intersect_samples_genes
from utils.data_interface import check_intersect_genes
from utils.data_interface import pca_components_selection
from utils.data_interface import plot_FactorGraph
from utils.data_interface import get_cycles
from utils.data_interface import save_CycleCollapsed_factors_nodes
from utils.data_interface import remove_outside_compounds
from utils.data_interface import remove_grad_files
from utils.data_interface import save_snn_model_weights


from scFEA.src.scFEA import scFEA
from MPO.mpo import mpo
from SNN.snn import snn

# global variables
SEP_SIGN = '*' * 100


def main(args):
    # pl.seed_everything(args.seed)
    
    # Init the output dir
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    else:
        os.makedirs(args.output_dir)
    
    #load gene expression data
    geneExpression = load_geneExpression(args)#geneExpression is the gene expression data, cols:=samples/cells, rows:=genes
    
    # load the modules(reactions) and the contained genes
    modules_genes = load_modulesGenes(args)
    all_modules = list(modules_genes.keys())
    all_modules.sort()
    
    #load the adjacency matrix of the factor graph
    compounds_modules = load_module(args)#compouns_modules is the adj matrix of the factor graph (reaction graph), rows:=compounds, columns:=modules
    compounds_modules=compounds_modules[all_modules]
    
    #the detection of cycles in the factor graph
    title_name="Compounds_Modules_FactorGraph_original"
    BDG=plot_FactorGraph(compounds_modules,title_name,args)#bipartite directed graph
    
    #sys.exit(1)
    
    cycles_in_graph=get_cycles(BDG)
    
    if len(cycles_in_graph)>0:
        BDG,compounds_modules=save_CycleCollapsed_factors_nodes(compounds_modules,BDG,cycles_in_graph,args)
    else:
        print("\n Non Cycle! \n")
    compounds_modules = remove_outside_compounds(compounds_modules)
    
    #sys.exit()
    
    # remove non overlap genes
    geneExpression = intersect_samples_genes(geneExpression,modules_genes)
    #std_mean_col_geneExpression = np.mean(np.std(geneExpression.T.div(geneExpression.max(axis=0), axis=0).T,axis=0))
    
     # print the selected genes
    check_intersect_genes(geneExpression.columns.values,modules_genes)
    
    # components selection by pca, the default is to keep 90% of information
    geneExpression_pca,modules_genes_pca=None,None
    if args.pca_components_selection==1:
        geneExpression_pca,modules_genes_pca=pca_components_selection(geneExpression,modules_genes,n_components=0.9)
        geneExpression=geneExpression_pca
        modules_genes=modules_genes_pca
        print("\n modules_genes_pca: \n {0} \n".format(modules_genes))
        print("\n geneExpression_pca: \n {0} \n".format(geneExpression))
    
    #sys.exit(1)
    
    
    # no training, only predicting by saved supervised Neural Network Parameters
    if args.do_train_snn==0 and args.do_predict_snn==1:
        print("\n Predicting Only....... \n")
        samples_modules_snn = snn(geneExpression, modules_genes, [], 0, args)
        samples_modules_snn = samples_modules_snn.abs()
        samples_modules_snn = samples_modules_snn[all_modules]
        samples_modules_snn.index = geneExpression.index
        samples_modules_snn = samples_modules_snn.T.div(samples_modules_snn.max(axis=0), axis=0).T
        samples_modules_snn.to_csv(args.output_dir+'flux_snn.csv',index=True,header=True)
        #print("\n samples_modules_snn:\n {0} \n".format(samples_modules_snn))
        
        # 2nd step to improve the initial values by Massage Passing Optimizer
        samples_modules_mpo = mpo(compounds_modules.copy(), samples_modules_snn.copy(), args)
        samples_modules_mpo = samples_modules_mpo[all_modules]
        samples_modules_mpo.index = geneExpression.index
        samples_modules_mpo = samples_modules_mpo.T.div(samples_modules_mpo.max(axis=0), axis=0).T
        samples_modules_mpo.to_csv(args.output_dir+'flux_snn_mpoOptimized.csv',index=True,header=True)
        #print("\n Massage Passing Optimazer Optimization Done! \n")

        print("\n Prediction Done! \n")
        return True
    

    #all steps: scfea->bp<->snn(training,testing,predicting)

    std_mean_col_scfea_all = []
    std_mean_col_mpo_all = []
    std_mean_col_snn_all = []

    scale_mean_all_scfea_all = []
    scale_mean_all_mpo_all = []
    scale_mean_all_snn_all = []

    imbalanceLoss_mean_scfea_all = []
    imbalanceLoss_mean_mpo_all = []
    imbalanceLoss_mean_snn_all = []
    
    
    flux_scfea_final = None
    flux_scfea_mpo_final = None
    flux_snn_final = None
    flux_snn_mpo_final = None
    
    
    #init the outputs dir
    tmp_flux_res_save_dir =  args.output_dir + 'tmp_flux_res/'
    if os.path.exists(tmp_flux_res_save_dir):
        shutil.rmtree(tmp_flux_res_save_dir)
        os.makedirs(tmp_flux_res_save_dir)
    else:
        os.makedirs(tmp_flux_res_save_dir)
        
    snn_final_weights_dir = "./src/SNN/final_model_weights/"
    if os.path.exists(snn_final_weights_dir):
        shutil.rmtree(snn_final_weights_dir)
        os.makedirs(snn_final_weights_dir)
    else:
        os.makedirs(snn_final_weights_dir)
        
    snn_history_weights_dir = "./src/SNN/model_weights_checkpoints/"
    if os.path.exists(snn_history_weights_dir):
        shutil.rmtree(snn_history_weights_dir)
        os.makedirs(snn_history_weights_dir)
    else:
        os.makedirs(snn_history_weights_dir)

    #########################################################################################################################

    # 1st step to generate the initial values by scFEA
    samples_modules_scfea = scFEA(geneExpression, modules_genes, compounds_modules, args)
    #samples_modules_scfea = samples_modules_scfea[compounds_modules.columns.values]
    samples_modules_scfea.index = geneExpression.index
    samples_modules_scfea.columns = all_modules
    flux_scfea_final = samples_modules_scfea.copy()
    save_samples_modules(samples_modules_scfea,'scfea',0,args)
    print("\n scFEA Done! \n")
    #sys.exit(1)

    #########################################################################################################################
    samples_modules_init = samples_modules_scfea.copy()
    cur_step = 0
    save_flux_snn_final_threshold = float('-inf')
    max_snn_step = -1
    cur_imbalanceLoss_mean_snn_all = float('inf')
    samples_modules_mpo_loss_min_target = []
    loss_mean_min_mpo = float('inf')
    while (cur_step < args.epoch_limit_all) or cur_imbalanceLoss_mean_snn_all < args.imbalance_loss_limit_all:
        print(SEP_SIGN)
        print("\n The {0}th epoch_all! \n".format(cur_step + 1))
        print(SEP_SIGN)

        #########################################################################################################################
        # 2nd step to improve the initial values by Massage Passing Optimizer
        samples_modules_mpo = mpo(compounds_modules.copy(), samples_modules_init.copy(), args)
        samples_modules_mpo = samples_modules_mpo[all_modules]
        samples_modules_mpo.index = geneExpression.index
        mpo_save_name = 'snn_labels_mpo'
        if cur_step==0:
            mpo_save_name = 'scfea_mpoOptimized'
            flux_scfea_mpo_final = samples_modules_mpo.copy()
        save_samples_modules(samples_modules_mpo,mpo_save_name, cur_step, args)
        print("\n Massage Passing Optimazer Optimization Done! \n")
        # print("\n samples_modules_mpo:\n {0} \n".format(samples_modules_mpo.iloc[:5, :]))
        #sys.exit(1)
        #########################################################################################################################

        # 3rd step to improve the flux values by supervised neuron network
        if len(samples_modules_mpo_loss_min_target) == 0:
            samples_modules_mpo_loss_min_target = samples_modules_mpo.copy()

        samples_modules_snn = snn(geneExpression.copy(), modules_genes.copy(), samples_modules_mpo_loss_min_target.copy(), cur_step+1, args)
        
        cur_step += 1
        
        if len(samples_modules_snn)==0:
            remove_grad_files(max_snn_step,args)
            continue
        samples_modules_snn = samples_modules_snn[all_modules]
        samples_modules_snn.index = geneExpression.index
        save_samples_modules(samples_modules_snn,'snn', cur_step, args)
        print("\n Current Supervised Neural Network Done! \n")
        # print("\n samples_modules_snn:\n {0} \n".format(samples_modules_snn.iloc[:5, :]))

        print("\n samples_modules_scfea:\n {0} \n".format(samples_modules_scfea))
        print("\n samples_modules_mpo:\n {0} \n".format(samples_modules_mpo))
        print("\n samples_modules_snn:\n {0} \n".format(samples_modules_snn))
        
        #sys.exit(1)
        #########################################################################################################################

        samples_modules_init = samples_modules_snn.copy()
        

        std_scale_imbalanceLoss = get_std_scale_imbalanceLoss_realData(compounds_modules,
                                                                       samples_modules_scfea,
                                                                       samples_modules_mpo_loss_min_target,
                                                                       samples_modules_snn)
        # print("\n std_mse_imbalanceLoss: \n {0} \n".format(std_scale_imbalanceLoss))
        # print("\n real std:{0} \n".format(std_mean_col_solutions))

        std_mean_col_scfea_all.append(std_scale_imbalanceLoss['std_mean_col_scfea'])
        std_mean_col_mpo_all.append(std_scale_imbalanceLoss['std_mean_col_mpo'])
        std_mean_col_snn_all.append(std_scale_imbalanceLoss['std_mean_col_snn'])

        scale_mean_all_scfea_all.append(std_scale_imbalanceLoss['scale_mean_all_scfea'])
        scale_mean_all_mpo_all.append(std_scale_imbalanceLoss['scale_mean_all_mpo'])
        scale_mean_all_snn_all.append(std_scale_imbalanceLoss['scale_mean_all_snn'])

        imbalanceLoss_mean_scfea_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_scfea'])
        imbalanceLoss_mean_mpo_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_mpo'])
        imbalanceLoss_mean_snn_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_snn'])

        # cur_mean_std_mpo=cos_std_imbalanceLoss['std_mean_mpo']
        samples_modules_mpo_tmp = samples_modules_mpo.copy()
        samples_modules_mpo_tmp = samples_modules_mpo_tmp.T.div(samples_modules_mpo_tmp.max(axis=0), axis=0).T
        cur_mean_loss_mpo = np.mean(get_imbalanceLoss(compounds_modules, samples_modules_mpo_tmp.values))
        print("\n cur_mean_loss_mpo:{0} \n".format(cur_mean_loss_mpo))
        print("\n loss_mean_min_mpo:{0} \n".format(loss_mean_min_mpo))

        #if cur_mean_loss_mpo < loss_mean_min_mpo:
        if True:
            samples_modules_mpo_loss_min_target = samples_modules_mpo.copy()
            loss_mean_min_mpo = cur_mean_loss_mpo

        #########################################################################################################################

        # sys.exit(1)

        # plot the figures
        cur_title_end = ""
        cur_activation_function = "TanhShrinkage"
        cur_title_end = cur_title_end + cur_activation_function
        plot_std_scale_imbalance_in_one(std_mean_col_scfea_all, std_mean_col_mpo_all, std_mean_col_snn_all,
                                        scale_mean_all_scfea_all, scale_mean_all_mpo_all, scale_mean_all_snn_all,
                                        imbalanceLoss_mean_scfea_all, imbalanceLoss_mean_mpo_all,
                                        imbalanceLoss_mean_snn_all,
                                        cur_title_end, args)
        # sys.exit(1)
        
        if std_scale_imbalanceLoss['std_mean_col_snn'] > save_flux_snn_final_threshold:
            max_snn_step = cur_step
            flux_snn_final = samples_modules_snn
            flux_snn_mpo_final = samples_modules_mpo
            save_flux_snn_final_threshold = std_scale_imbalanceLoss['std_mean_col_snn']
        
        print("\n max_snn_step={0} \n".format(max_snn_step))
        remove_grad_files(max_snn_step,args)
            

    print("\n max_snn_step={0} \n".format(max_snn_step))
    save_snn_model_weights(max_snn_step,args)
    
    
    # save final flux res
    flux_scfea_final.to_csv(args.output_dir+'flux_scfea.csv',index=True,header=True)
    flux_scfea_mpo_final.to_csv(args.output_dir+'flux_scfea_mpo.csv',index=True,header=True)
    flux_snn_final.to_csv(args.output_dir+'flux_snn.csv',index=True,header=True)
    flux_snn_mpo_final.to_csv(args.output_dir+'flux_snn_mpo.csv',index=True,header=True)


def parse_arguments(parser):
    # global parameters
    # parser.add_argument("--seed", default=2345, type=int)
    parser.add_argument('--input_dir', type=str, default="./inputs/",
                        help="The inputs directory.")
    parser.add_argument('--output_dir', type=str, default="./outputs/",
                        help="The outputs directory, you can find all outputs in this directory.")
    parser.add_argument('--geneExpression_file_name', type=str, default='geneExpression_test.csv.gz',
                        help="The scRNA-seq file name.")
    parser.add_argument('--compounds_modules_file_name', type=str, default='compouns_modules_test.csv',
                        help="The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder.")
    parser.add_argument('--modules_genes_file_name',type=str,default='modules_genes_test.json',
                        help="The json file contains genes for each module. We provide human and mouse two models in scFEA. For human model, please use module_gene_m168.csv which is default.  All candidate moduleGene files are provided in /data/ folder.")
    parser.add_argument('--cycle_detection',type=int,default=1,
                        help="Remove the cycles in the graph. 0=False, 1=True")
    parser.add_argument('--epoch_limit_all', type=int, default=20,
                        help="The user defined early stop Epoch(the whole framework)")
    parser.add_argument('--imbalance_loss_limit_all', type=float, default=0.01,
                        help="The user defined early stop imbalance loss.")
    parser.add_argument('--save_predictions', type=int, default=1,
                        help="Save results. 0=False, 1=True")
    parser.add_argument('--pca_components_selection',type=int, default=0,
                        help="Apply PCA to reduce the dimension of features. 0=False, 1=True")


    # parameters for scFEA
    parser.add_argument('--n_epoch_scfea', type=int, default=100,
                        help="User defined Epoch for scFEA training.")

    # parameters for bp_balance
    parser.add_argument('--n_epoch_mpo', type=int, default=30,
                        help="User defined Epoch for Message Passing Optimizer.")

    # parameters for snn
    parser.add_argument('--n_epoch_snn', type=int, default=200,
                        help="User defined Epoch for Supervised Neural Network training.")
    parser.add_argument('--do_train_snn', type=int, default=1,
                        help="Train the SNN model, 0=False, 1=True.")
    parser.add_argument('--n_train_batch_snn',type=int, default=128)
    parser.add_argument('--do_predict_snn', type=int, default=1,
                        help="Predict the flux values via the trained SNN model, 0=False, 1=True. FYI: If you have already trained the SNN model, SNN saves the model automatically, then you can set --do_train_snn 0 and --do_predict_snn 1 to predict the flux values directly.")
    parser.add_argument('--output_grad_snn',type=int,default=1,
                        help="Save the gradients on each gene.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # mp.set_start_method('spawn',force=True)
    parser = argparse.ArgumentParser(description='MPOSNN: A Massage Passing Optimizer-Based Supervised Neural Network Model to Estimate Cell-Wise Metabolic Using Single Cell RNA-seq Data.')

    # global args
    args = parse_arguments(parser)

    main(args)
