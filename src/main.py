# -*-coding:utf-8-*-


# build-in libraries
import sys
import argparse
import warnings
import magic

# Third-party libraries
import numpy as np
import pandas as pd

# my libraries
from utils.data_interface import load_module
from utils.data_interface import load_geneExpression
from utils.data_interface import load_modulesGenes
from utils.data_interface import save_samples_modules
from utils.data_interface import get_std_scale_imbalanceLoss_realData
from utils.data_interface import get_imbalanceLoss
from utils.data_interface import plot_std_scale_imbalance_in_one
from utils.data_interface import intersect_samples_genes
from utils.data_interface import pca_components_selection
from utils.data_interface import remove_grad_files
from utils.data_interface import save_snn_model_weights
from utils.data_interface import save_grad_file
from utils.data_interface import update_compoundsModules_modulesGenes
from utils.data_interface import init_dir
from utils.data_interface import plot_FactorGraph
from utils.data_interface import get_output_path
from sklearn import preprocessing

from scFEA.src.scFEA import scFEA
from MPO_LoopOrAnchor.mpo import mpo
from SNN.snn import snn

# global variables
SEP_SIGN = '*' * 100


def main(args):
    # pl.seed_everything(args.seed)

    print(SEP_SIGN)
    print("Cur Input parameters:\n{0}\n".format(args))
    print(SEP_SIGN)

    # Init the output dir
    output_folder = None
    if args.do_train_snn == 1 and args.do_predict_snn == 1:
        args.output_dir, output_folder = get_output_path(args)
        args.load_checkpoints_dir = args.output_dir

        # make output dir if not exist
        init_dir(args.output_dir)

        # init the outputs dir
        tmp_flux_res_save_dir = args.output_dir + 'tmp_flux_res/'
        init_dir(tmp_flux_res_save_dir)

        # snn_final_weights_dir = "./src/SNN/final_model_weights/"
        snn_final_weights_dir = args.output_dir + "SNN/final_model_weights/"
        init_dir(snn_final_weights_dir)

        # snn_history_weights_dir = "./src/SNN/model_weights_checkpoints/"
        snn_history_weights_dir = args.output_dir + "SNN/model_weights_checkpoints/"
        init_dir(snn_history_weights_dir)

    # load gene expression data
    geneExpression = load_geneExpression(
        args)  # geneExpression is the gene expression data, cols:=samples/cells, rows:=genes, but the data will be transposed to rows:=samples/cells, cols:=genes automatically

    # load the modules(reactions) and the contained genes
    modules_genes = load_modulesGenes(args)

    # remove non overlap genes
    geneExpression, modules_genes = intersect_samples_genes(geneExpression, modules_genes)
    if len(geneExpression) == 0:
        print("\n No Intersection of Genes between Data and (Modules)Reactions! \n")
        return False

    # load the adjacency matrix of the factor graph
    compounds_modules = load_module(
        args)  # compouns_modules is the adj matrix of the factor graph (reaction graph), rows:=compounds, columns:=modules

    # intersection of modules in compounds_modules and modules_genes
    compounds_modules, modules_genes, geneExpression = update_compoundsModules_modulesGenes(compounds_modules,
                                                                                            modules_genes,
                                                                                            geneExpression)

    # plot the graph again after removing invalid modules and compounds
    if args.do_train_snn == 1 and args.do_predict_snn == 1:
        plot_FactorGraph(compounds_modules, title_name='compounds_modules_connected',
                         save_path=args.output_dir + 'compounds_modules_connected.png')

    # imputation if too many missing values
    if args.do_imputation == 1:
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpression = magic_operator.fit_transform(geneExpression)

    # pca_components_selection, the default is 0, which means no pca_components_selection
    if args.pca_components_selection == 1:
        geneExpression_pca, modules_genes_pca = None, None
        geneExpression_pca, modules_genes_pca = pca_components_selection(geneExpression, modules_genes,
                                                                         n_components=0.9)
        geneExpression = geneExpression_pca
        modules_genes = modules_genes_pca
        # print("\n modules_genes_pca: \n {0} \n".format(modules_genes))
        # print("\n geneExpression_pca: \n {0} \n".format(geneExpression))

    # normalize the gene expression data
    scaler = preprocessing.MinMaxScaler()
    geneExpr_scaled = scaler.fit_transform(geneExpression.values.copy())
    geneExpression = pd.DataFrame(geneExpr_scaled, index=geneExpression.index, columns=geneExpression.columns)

    # no training, only predicting by saved supervised Neural Network Parameters
    if args.do_train_snn == 0 and args.do_predict_snn == 1:
        print("\n Predicting Only....... \n")

        args.load_checkpoints_dir = args.output_dir + args.load_weights_folder + '/'
        cur_data_name = args.geneExpression_file_name.split('.csv')[0]
        cur_module_source = args.module_source

        if cur_data_name + '-' + cur_module_source in args.load_weights_folder:
            args.output_dir = args.output_dir + args.load_weights_folder + '_PredictionOnly/'
            init_dir(args.output_dir)
        else:
            args.output_dir = args.output_dir + cur_data_name + '-' + cur_module_source + '_PredictionOnly-' + args.load_weights_folder + '/'
            init_dir(args.output_dir)

        samples_modules_snn = snn(geneExpression.copy(), modules_genes.copy(), [], 0, args)
        if len(samples_modules_snn) == 0:
            print("Prediction Failed! \n")
            return False

        # samples_modules_snn = min_max_normalization(samples_modules_snn, by="col")
        samples_modules_snn.to_csv(args.output_dir + 'flux_snn.csv', index=True, header=True)
        # print("\n samples_modules_snn:\n {0} \n".format(samples_modules_snn))

        # 2nd step to improve the initial values by Massage Passing Optimizer
        samples_modules_mpo = mpo(compounds_modules.copy(), samples_modules_snn.copy(), args)
        samples_modules_mpo.to_csv(args.output_dir + 'flux_snn_mpoOptimized.csv', index=True, header=True)
        print("\n Massage Passing Optimazer Optimization Done! \n")
        save_grad_file(args)
        print("\n Prediction Done! \n")

        print("Your Results are saved in: {0}\n".format(args.output_dir))
        return True

    # all steps: scfea->mpo<->snn(training,testing,predicting)

    std_mean_col_scfea_all = []
    std_mean_col_mpo_all = []
    std_mean_col_snn_all = []
    std_mean_col_mposnn_all = []

    scale_mean_all_scfea_all = []
    scale_mean_all_mpo_all = []
    scale_mean_all_snn_all = []
    scale_mean_all_mposnn_all = []

    imbalanceLoss_mean_scfea_all = []
    imbalanceLoss_mean_mpo_all = []
    imbalanceLoss_mean_snn_all = []
    imbalanceLoss_mean_mposnn_all = []

    flux_scfea_final = None
    flux_scfea_mpo_final = None
    flux_snn_final = None
    flux_snn_mpo_final = None
    flux_mposnn_final = []

    #########################################################################################################################

    # 1st step to generate the initial values by scFEA
    samples_modules_scfea = scFEA(geneExpression.copy(), modules_genes.copy(), compounds_modules.copy(), args)
    samples_modules_scfea.index = geneExpression.index
    samples_modules_scfea.columns = compounds_modules.columns
    # print("\n samples_modules_scfea:\n {0} \n".format(samples_modules_scfea))
    flux_scfea_final = samples_modules_scfea.copy()

    # save the initial fluxes
    save_samples_modules(samples_modules_scfea.copy(), 'scfea', 0, args)
    print("\n scFEA Done! \n")

    #########################################################################################################################
    samples_modules_init = samples_modules_scfea.copy()
    cur_step = 0

    # threshold to choose snn's result
    minImbalance_flux_snn = float('inf')
    maxStd_flux_snn = float('-inf')
    minImbalanceMaxStd_flux_snn_epoch = -1
    cur_imbalanceLoss_mean_snn_all = float('inf')
    cur_std_mean_snn_all = float('-inf')

    # threshold to update mpo's result as label
    minImbalance_flux_mpo = float('inf')
    maxStd_flux_mpo = float('-inf')

    samples_modules_mpo_loss_min_target = []
    # loss_mean_min_mpo = float('inf')
    while (cur_step < args.n_epoch_all) or (
            cur_imbalanceLoss_mean_snn_all < args.imbalance_loss_limit_all and cur_std_mean_snn_all > 10):
        print(SEP_SIGN)
        print("\n The {0}th epoch_all! \n".format(cur_step + 1))
        print(SEP_SIGN)

        #########################################################################################################################
        # 2nd step to improve the initial values by Massage Passing Optimizer
        print("\n Massage Passing Optimazer Optimization Running! \n")
        samples_modules_mpo = mpo(compounds_modules.copy(), samples_modules_init.copy(), args)
        mpo_save_name = 'snn_labels_mpo'
        if cur_step == 0:
            mpo_save_name = 'scfea_mpoOptimized'
            flux_scfea_mpo_final = samples_modules_mpo.copy()
            times = 1000.0
            samples_modules_mpo_tmp = samples_modules_mpo.div(samples_modules_mpo.sum(axis=1), axis=0) * times
            minImbalance_flux_mpo = np.mean(get_imbalanceLoss(compounds_modules, samples_modules_mpo_tmp.values))
            maxStd_flux_mpo = np.mean(np.std(samples_modules_mpo_tmp.values, axis=0))

        save_samples_modules(samples_modules_mpo.copy(), mpo_save_name, cur_step, args)
        print("\n current Massage Passing Optimazer Optimization Done! \n")
        # print("\n samples_modules_mpo:\n {0} \n".format(samples_modules_mpo))
        # sys.exit(1)
        #########################################################################################################################

        # 3rd step to improve the flux values by supervised neuron network
        if len(samples_modules_mpo_loss_min_target) == 0:
            samples_modules_mpo_loss_min_target = samples_modules_mpo.copy()

        # samples_modules_mpo_loss_min_target = min_max_normalization(samples_modules_mpo_loss_min_target, by='col')
        samples_modules_snn = snn(geneExpression.copy(), modules_genes.copy(),
                                  samples_modules_mpo_loss_min_target.copy(), cur_step + 1, args)


        cur_step += 1

        if len(samples_modules_snn) == 0:
            remove_grad_files(minImbalanceMaxStd_flux_snn_epoch, args)
            continue

        save_samples_modules(samples_modules_snn, 'snn', cur_step, args)
        print("\n Current Supervised Neural Network Done! \n")


        if len(flux_mposnn_final) == 0:
            flux_mposnn_final = (samples_modules_mpo+samples_modules_snn+samples_modules_mpo_loss_min_target)/3.0
        else:
            flux_mposnn_final = (flux_mposnn_final+samples_modules_mpo+samples_modules_snn+samples_modules_mpo_loss_min_target)/4.0

        print("\n samples_modules_scfea:\n {0} \n".format(samples_modules_scfea))
        print("\n samples_modules_mpo:\n {0} \n".format(samples_modules_mpo))
        print("\n samples_modules_snn:\n {0} \n".format(samples_modules_snn))

        #########################################################################################################################

        samples_modules_init = samples_modules_snn.copy()

        std_scale_imbalanceLoss = get_std_scale_imbalanceLoss_realData(compounds_modules.copy(),
                                                                       samples_modules_scfea.copy(),
                                                                       samples_modules_mpo_loss_min_target.copy(),
                                                                       samples_modules_snn.copy(),
                                                                       flux_mposnn_final.copy())
        # print("\n std_mse_imbalanceLoss: \n {0} \n".format(std_scale_imbalanceLoss))
        # print("\n real std:{0} \n".format(std_mean_col_solutions))

        std_mean_col_scfea_all.append(std_scale_imbalanceLoss['std_mean_col_scfea'])
        std_mean_col_mpo_all.append(std_scale_imbalanceLoss['std_mean_col_mpo'])
        std_mean_col_snn_all.append(std_scale_imbalanceLoss['std_mean_col_snn'])
        std_mean_col_mposnn_all.append(std_scale_imbalanceLoss['std_mean_col_mposnn'])

        scale_mean_all_scfea_all.append(std_scale_imbalanceLoss['scale_mean_all_scfea'])
        scale_mean_all_mpo_all.append(std_scale_imbalanceLoss['scale_mean_all_mpo'])
        scale_mean_all_snn_all.append(std_scale_imbalanceLoss['scale_mean_all_snn'])
        scale_mean_all_mposnn_all.append(std_scale_imbalanceLoss['scale_mean_all_mposnn'])

        imbalanceLoss_mean_scfea_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_scfea'])
        imbalanceLoss_mean_mpo_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_mpo'])
        imbalanceLoss_mean_snn_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_snn'])
        imbalanceLoss_mean_mposnn_all.append(std_scale_imbalanceLoss['imbalanceLoss_mean_mposnn'])

        cur_imbalanceLoss_mean_snn_all = std_scale_imbalanceLoss['imbalanceLoss_mean_snn']
        cur_std_mean_snn_all = std_scale_imbalanceLoss['std_mean_col_snn']

        # update the labels from mpo
        # if True:
        if std_scale_imbalanceLoss['std_mean_col_mpo'] > maxStd_flux_mpo and std_scale_imbalanceLoss[
            'imbalanceLoss_mean_mpo'] < minImbalance_flux_mpo:
            samples_modules_mpo_loss_min_target = samples_modules_mpo.copy()
            maxStd_flux_mpo = std_scale_imbalanceLoss['std_mean_col_mpo']
            minImbalance_flux_mpo = std_scale_imbalanceLoss['imbalanceLoss_mean_mpo']

        #########################################################################################################################

        # sys.exit(1)

        # plot the figures
        #cur_title_end = ""
        #cur_activation_function = "TanhShrinkage"
        #cur_title_end = cur_title_end + cur_activation_function
        '''
        plot_std_scale_imbalance_in_one(std_mean_col_scfea_all, std_mean_col_mpo_all, std_mean_col_snn_all, std_mean_col_mposnn_all,
                                        scale_mean_all_scfea_all, scale_mean_all_mpo_all, scale_mean_all_snn_all,scale_mean_all_mposnn_all,
                                        imbalanceLoss_mean_scfea_all, imbalanceLoss_mean_mpo_all,
                                        imbalanceLoss_mean_snn_all, imbalanceLoss_mean_mposnn_all,
                                        args)
        '''
        # sys.exit(1)

        if std_scale_imbalanceLoss['imbalanceLoss_mean_snn'] < minImbalance_flux_snn and std_scale_imbalanceLoss[
            'std_mean_col_snn'] > maxStd_flux_snn:
            minImbalanceMaxStd_flux_snn_epoch = cur_step
            flux_snn_final = samples_modules_snn
            flux_snn_mpo_final = samples_modules_mpo
            minImbalance_flux_snn = std_scale_imbalanceLoss['imbalanceLoss_mean_snn']
            maxStd_flux_snn = std_scale_imbalanceLoss['std_mean_col_snn']

        # print("\n min_snn_step={0} \n".format(minImbalance_flux_snn_epoch))
        remove_grad_files(minImbalanceMaxStd_flux_snn_epoch, args)


    # print("\n max_snn_step={0} \n".format(minImbalance_flux_snn_epoch))
    save_snn_model_weights(minImbalanceMaxStd_flux_snn_epoch, args)
    save_grad_file(args)

    # save final flux res
    # flux_scfea_final = flux_scfea_final.T.div(flux_scfea_final.max(axis=0), axis=0).T
    # flux_scfea_final = min_max_normalization(flux_scfea_final, by='col')
    flux_scfea_final.to_csv(args.output_dir + 'flux_scfea.csv', index=True, header=True)

    # flux_scfea_mpo_final = min_max_normalization(flux_scfea_mpo_final, by='col')
    flux_scfea_mpo_final.to_csv(args.output_dir + 'flux_scfea_mpo.csv', index=True, header=True)

    # flux_snn_final = flux_snn_final.T.div(flux_snn_final.max(axis=0), axis=0).T
    # flux_snn_final = min_max_normalization(flux_snn_final, by='col')
    #flux_snn_final.to_csv(args.output_dir + 'flux_snn.csv', index=True, header=True)

    # flux_snn_mpo_final = min_max_normalization(flux_snn_mpo_final, by='col')
    #flux_snn_mpo_final.to_csv(args.output_dir + 'flux_snn_mpo.csv', index=True, header=True)


    flux_mposnn_final.to_csv(args.output_dir + 'flux_mposnn.csv', index=True, header=True)

    print("All Done! \n")

    print(
        "Your Results are saved in: {0}\nPlease remember the folder name below, you need this name to do the prediction only task:\n{1}".format(
            args.output_dir, output_folder))


def parse_arguments(parser):
    # global parameters
    # parser.add_argument("--seed", default=2345, type=int)
    parser.add_argument('--input_dir', type=str, default="./inputs/",
                        help="The inputs directory.")
    parser.add_argument('--output_dir', type=str, default="./outputs/",
                        help="The outputs directory, you can find all outputs in this directory.")
    parser.add_argument('--geneExpression_file_name', type=str, default='NA',
                        help="The scRNA-seq file name.")
    parser.add_argument('--compounds_modules_file_name', type=str, default='NA',
                        help="The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder.")
    parser.add_argument('--modules_genes_file_name', type=str, default='NA',
                        help="The json file contains genes for each module. We provide human and mouse two models in scFEA.")
    parser.add_argument('--n_epoch_all', type=int, default=100,
                        help="The user defined early stop Epoch(the whole framework)")
    parser.add_argument('--imbalance_loss_limit_all', type=float, default=0.01,
                        help="The user defined early stop imbalance loss.")
    parser.add_argument('--save_predictions', type=int, default=1,
                        help="Save results. 0=False, 1=True")
    parser.add_argument('--pca_components_selection', type=int, default=0,
                        help="Apply PCA to reduce the dimension of features. 0=False, 1=True")
    parser.add_argument('--do_imputation', type=int, default=0,
                        help="Imputation on the input gene expression matrix. 0=False, 1=True")
    parser.add_argument('--experiment_name', type=str, default="FluxEstimation")
    parser.add_argument('--module_source', type=str, default="NA")
    parser.add_argument('--load_checkpoints_dir', type=str, default="NA")
    parser.add_argument('--load_weights_folder', type=str, default="NA")

    # parameters for scFEA
    parser.add_argument('--n_epoch_scfea', type=int, default=200,
                        help="User defined Epoch for scFEA training.")

    # parameters for bp_balance
    parser.add_argument('--n_epoch_mpo', type=int, default=50,
                        help="User defined Epoch for Message Passing Optimizer.")

    # parameters for snn
    parser.add_argument('--n_epoch_snn', type=int, default=300,
                        help="User defined Epoch for Supervised Neural Network training.")
    parser.add_argument('--do_train_snn', type=int, default=1,
                        help="Train the SNN model, 0=False, 1=True.")
    parser.add_argument('--n_train_batch_snn', type=int, default=1000000)
    parser.add_argument('--do_predict_snn', type=int, default=1,
                        help="Predict the flux values via the trained SNN model, 0=False, 1=True. FYI: If you have already trained the SNN model, SNN saves the model automatically, then you can set --do_train_snn 0 and --do_predict_snn 1 to predict the flux values directly.")
    parser.add_argument('--output_grad_snn', type=int, default=1,
                        help="Save the gradients on each gene.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # mp.set_start_method('spawn',force=True)
    parser = argparse.ArgumentParser(
        description='MPOSNN: A Massage Passing Optimizer-Based Supervised Neural Network Model to Estimate Cell-Wise Metabolic Using Single Cell RNA-seq Data.')

    # global args
    args = parse_arguments(parser)

    main(args)
