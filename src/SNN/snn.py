import os
import sys
#from multiprocessing import Pool
#from multiprocessing import cpu_count
# import pprint
import shutil

import pandas as pd
import numpy as np
#import pysnooper
import torch
#import pytorch_lightning as pl
#from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.multiprocessing import Process,Pool
from SNN.utils.data_interface import DataSet
from SNN.utils.data_interface import shuffle_geneExpression
from SNN.utils.model_interface import LitFCN



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#@pysnooper.snoop()
def get_imbalance_factors(samples_modules,n_samples,factors):
    samples_factors = {}

    if len(samples_modules)==0:
        for factor_i in factors:
            imbalance=[0]*n_samples
            samples_factors[factor_i]=imbalance
    else:
        tmp_col_zeros = np.zeros(n_samples)
        tmp_col_zeros = pd.DataFrame(tmp_col_zeros)
        tmp_col_zeros.index=samples_modules.index
        tmp_col_zeros.columns = ['tmp_zero']

        for factor_i in factors:
            if factor_i.startswith('dummy_'):
                imbalance = [0] * n_samples
                samples_factors[factor_i] = imbalance
                continue
            parent_nodes=cur_factor_graph._factors[factor_i]['parent_nodes']
            child_nodes=cur_factor_graph._factors[factor_i]['child_nodes']
            sum_parent_nodes=np.sum(pd.concat([samples_modules[parent_nodes],tmp_col_zeros],axis=1),axis=1)
            sum_child_nodes=np.sum(pd.concat([samples_modules[child_nodes],tmp_col_zeros],axis=1),axis=1)
            imbalance=sum_parent_nodes-sum_child_nodes
            samples_factors[factor_i]=imbalance.values

    samples_factors=pd.DataFrame.from_dict(samples_factors)
    return samples_factors




def save_supervisedNN_flux(data):
    save_file_dir = args.project_dir + args.res_dir + args.data_name + "_" + args.module_source + "_flux_supervisedNN.csv.gz"
    data = pd.DataFrame(data)
    data.index = samples_genes.index

    print(data)
    data.to_csv(save_file_dir, index=True, header=True, compression="gzip")
    print("\nPrediction Done! \n supervised NN flux csv.gz saved.")

    return True


def load_best_checkpoint_name(file_dir):
    ckpt_list=os.listdir(file_dir)
    #ckpt_list=[ckpt_file for ckpt_file in ckpt_list if ckpt_file.endswith("ckpt")]

    min_idx=0
    min_loss=float("inf")


    def ckpt2loss(ckpt_name):
        loss=ckpt_name.split("=")[-1][:-5]
        if "-v" in loss:
            loss=loss.split('-')[0]
        return float(loss)



    for i,ckpt_file in enumerate(ckpt_list):
        if not ckpt_file.endswith("ckpt"):
            continue

        cur_loss=ckpt2loss(ckpt_file)
        if cur_loss<min_loss:
            min_idx=i
            min_loss=cur_loss

    return ckpt_list[min_idx]



#@pysnooper.snoop()
def train_test(data,labels,module_name,cur_step,args):
    torch.set_num_threads(1)

    
    n_samples, n_cur_genes = data.shape
    args.n_train_batch_snn = min(args.n_train_batch_snn,n_samples)
    #args.n_train_batch_snn = n_samples
    data = torch.tensor(data.values).float()
    labels = torch.tensor(labels.values).float()

    data_set = DataSet(data, labels)

    training_set, validation_set, testing_set = random_split(data_set, [int(n_samples * 0.6),int(n_samples * 0.2), n_samples - int(n_samples * 0.6)-int(n_samples * 0.2)])
    training_loader = DataLoader(training_set, batch_size=args.n_train_batch_snn, shuffle=True, num_workers=0)
    validation_loader=DataLoader(validation_set,batch_size=n_samples, shuffle=False,num_workers=0)
    testing_loader = DataLoader(testing_set, batch_size=len(testing_set), num_workers=0)

    save_checkpoint_dir = "./src/SNN/model_weights_checkpoints/Epoch_"+str(cur_step).zfill(3)+'/'+module_name+'/'

    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
    else:
        #os.system('rm -rf '+save_checkpoint_dir)
        shutil.rmtree(save_checkpoint_dir)
        os.makedirs(save_checkpoint_dir)

    model = LitFCN(dim_in=n_cur_genes)

    #early_stopping_callback = EarlyStopping(monitor='train_loss', patience=2)
    checkpoint_callback_loss=ModelCheckpoint(
        dirpath=save_checkpoint_dir,
        #filename="best_checkpoint",
        verbose=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.4f}",
    )

    print("Training.....\n")
    trainer = Trainer(max_epochs=int(args.n_epoch_snn),
                      default_root_dir=save_checkpoint_dir,
                      logger=False,
                      callbacks=[checkpoint_callback_loss],
                      accelerator="cpu",
                      #checkpoint_callback=False,
                      )
    trainer.fit(model, training_loader,validation_loader)
    print("Training Done!\n")

    print("Testing.......\n")
    trainer.test(model, testing_loader, ckpt_path="best")
    print("Testing Done!\n")

    return True


def predict(data,module_name,cur_step,args):

    n_samples, n_cur_genes = data.shape

    print("Calculating the predictions and grad...........\n")
    
    #if args.output_grad_snn==1:
    data_grad = torch.tensor(data.values.reshape(n_samples*n_cur_genes),dtype=torch.float,requires_grad=True).reshape(n_samples,n_cur_genes)
    data_grad.retain_grad()
    read_checkpoint_dir=''
    if args.do_train_snn==1 and args.do_predict_snn==1:
        read_checkpoint_dir = "./src/SNN/model_weights_checkpoints/Epoch_"+str(cur_step).zfill(3)+'/'+module_name+"/"
    else:
        read_checkpoint_dir = "./src/SNN/final_model_weights/"+module_name+"/"
    
    if not os.path.exists(read_checkpoint_dir):
        print("No Checkpoints!!"+module_name+"!!"+read_checkpoint_dir)
        return module_name,-1
    
    checkpoint_name=load_best_checkpoint_name(read_checkpoint_dir)

    model = LitFCN.load_from_checkpoint(read_checkpoint_dir+checkpoint_name,dim_in=n_cur_genes)
    model.eval()
    model.freeze()
    

    samples_curModule = model.predict_step(data_grad)

    samples_curGenes_grad = []
    #if args.output_grad_snn==1:
    for i in range(samples_curModule.size(0)):
        torch.autograd.backward(samples_curModule[i][0],retain_graph=True, create_graph=False)
        samples_curGenes_grad.append(data_grad.grad[i,:].cpu().detach().numpy().copy())
        data_grad.grad.zero_()

    print("Module Name {0} Predicting and Grad Done!\n".format(module_name))
    
    samples_curModule = np.concatenate(samples_curModule.cpu().detach().numpy())
    samples_curGenes_grad=pd.DataFrame(np.vstack(samples_curGenes_grad),index=data.index,columns=data.columns)
    #sys.exit(1)
    
    return module_name, samples_curModule,samples_curGenes_grad


#@pysnooper.snoop()
def snn(geneExpression, modules_genes,samples_modules_mpo,cur_step, args):
    #pl.seed_everything(cur_args.seed)

    #global samples_genes
    samples_genes=geneExpression
    n_samples = geneExpression.shape[0]

    #global samples_modules_mpo_target
    samples_modules_mpo_target = samples_modules_mpo
    #print("\n samples_modules_mpo_target:\n {0} \n".format(samples_genes.iloc[:3,:3]))

    #global modules_genes_train
    modules_genes_train=modules_genes
    n_modules = len(modules_genes)
    
    all_model_genes = []
    for genes in modules_genes_train.values():
        all_model_genes.extend(genes)
    all_geneExpr_genes = set(samples_genes.columns.values)
    unique_genes = list(set(all_model_genes).intersection(all_geneExpr_genes))
    samples_genes=samples_genes[unique_genes]
    samples_genes = samples_genes.div((samples_genes.max(axis=1) - samples_genes.min(axis=1)), axis=0)
    #samples_genes=(samples_genes+1).apply(np.log2)

    ##################################################################################
    # predicting only
    samples_modules_snn = {}
    # predicting step
    #samples_modules_snn = {}
    if args.do_train_snn==0 and args.do_predict_snn==1:
        modules=list(modules_genes.keys())
        modules.sort()

        print("Predicting Only....!")
        try:
        #if True:
            for module_name in modules:
                cur_genes = modules_genes_train[module_name]
                cur_genes = list(set(cur_genes).intersection(set(samples_genes.columns.values)))
                if len(cur_genes)==0:
                    #res.append([module_name,-1000])
                    continue

                data = samples_genes[cur_genes].copy()  # row:sample(cell) col:genes

                module_name,samples_curModule,samples_curGenes_grad=predict(data,module_name,0,args)
                samples_modules_snn[module_name] = samples_curModule.copy()
                #if len(samples_curGenes_grad)!=0:
                grad_save_path = args.output_dir+'flux_snn_'+module_name+'_grad_'+str(cur_step).zfill(3)+'.csv'
                samples_curGenes_grad.to_csv(grad_save_path)
                #sys.exit(1)

            #print(samples_modules)
            samples_modules_snn=pd.DataFrame.from_dict(samples_modules_snn)
            samples_modules_snn.index=geneExpression.index
            samples_modules_snn=samples_modules_snn.abs()
        except:
            print("\n Prediction Error!\n")
        #sys.exit(1)
        return samples_modules_snn
    
    
    ##################################################################################
    # all steps training->testing->predicting
     # training step
    if args.do_train_snn==1:
        print("Training and Testing....!\n")

        save_checkpoint_dir = "./src/SNN/model_weights_checkpoints/"+"Epoch_"+str(cur_step).zfill(3)+'/'
        if os.path.exists(save_checkpoint_dir):
            shutil.rmtree(save_checkpoint_dir)

        #'''
        process_list=[]
        for module_name in samples_modules_mpo.columns.values:
                
            cur_genes = modules_genes_train[module_name]
            cur_genes = list(set(cur_genes).intersection(set(samples_genes.columns.values)))
                
            if len(cur_genes)==0:
                    continue
                
            data = samples_genes[cur_genes].copy()

            labels = samples_modules_mpo_target.loc[data.index.values, module_name]
            p=None
            p=Process(target=train_test,args=(data,labels,module_name,cur_step,args,))
            p.start()
            process_list.append(p)
        for p_i in process_list:
            p_i.join()
        #'''
        
    # predicting step
    #samples_modules_snn = {}
    if args.do_predict_snn==1:
        print("Predicting....!")
        try:
        #if True:
            for module_name in samples_modules_mpo.columns.values:
                cur_genes = modules_genes_train[module_name]
                cur_genes = list(set(cur_genes).intersection(set(samples_genes.columns.values)))

                if len(cur_genes)==0:
                    #res.append([module_name,-1000])
                    continue
                
                data = samples_genes[cur_genes].copy()  # row:sample(cell) col:genes
                module_name,samples_curModule,samples_curGenes_grad=predict(data,module_name,cur_step,args)
                samples_modules_snn[module_name] = samples_curModule.copy()
                #if len(samples_curGenes_grad)!=0:
                grad_save_path = args.output_dir+'flux_snn_'+module_name+'_grad_'+str(cur_step).zfill(3)+'.csv'
                samples_curGenes_grad.to_csv(grad_save_path)


            #print(samples_modules)
            samples_modules_snn=pd.DataFrame.from_dict(samples_modules_snn)
            samples_modules_snn.index=samples_modules_mpo_target.index
            samples_modules_snn=samples_modules_snn.abs()
        except:
            print("\n Prediction Error! \n")
            
    print("all Done!!!")
    n_res = len(samples_modules_snn)
    if n_samples!=n_res:
        return []
    return samples_modules_snn.abs()