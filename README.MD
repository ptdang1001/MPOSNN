##  MPOSNN: A Massage Passing Optimizer-Based Supervised Neural Network Model to Estimate Cell-Wise Metabolic Using Single Cell RNA-seq Data.

## Change Log

### v0.5
- Release the initial version and installtion manual
- Release human complete metabolic flux modules


### To be released soon 
If you have an interested topic, please feel free to open an [issue](https://github.com/ptdang1001/MPOSNN/issues) or I can also merge your completed function into main branch




## Introduction

>We developed a Message Passing Optimizer-Based Supervised Neural Network to Estimate the Cell-Wise Metabolic Using Single Cell RNA-seq Data. To infer single cell fluxome from single cell RNA-sequencing (scRNA-seq) data, Our framework is powered by three sub-algorithms:(1) scFEA(single cell Flux Estimation Analysis), a self-constrained Graph Neural Network to generate the initial flux values; (2) MPO(Message Passing Optimizer), a Belief Propagation-based message passing algorithm to optimize the initial flux values; (3) SNN(Supervised Neural Network), a neural network-based supervised learning algorithm to learn the predictor between the optimized flux values and the scRNA-seq data.

## The computational framework of MPOSNN

<p align="center">
  <img width="80%" src="https://github.com/ptdang1001/MPOSNN/blob/main/inputs/mposnn.png">
</p>

## Requirements and Installation

MPOSNN is implemented by Python3. If you don't have python, please download [Anaconda](https://www.anaconda.com/download/#linux) with python 3 version.

- torch >= 1.13.1
- numpy >= 1.23.3
- pandas >= 1.4.4
- matplotlib >=3.6.2
- magic >= 2.0.4
- scikit-learn >= 1.1.1
- networkx >= 2.8.8
- pytorch-lightning >= 1.8.1

Download MPOSNN:
```
git clone https://github.com/ptdang1001/MPOSNN
```

Install requirements:
```
cd MPOSNN
conda install --file requirements
conda install pytorch torchvision -c pytorch
pip install --user magic-impute
```

## Usage

You can see the input arguments for MPOSNN by help option:
```
python src/main.py --help
usage: src/main.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--geneExpression_file_name GENEEXPRESSION_FILE_NAME] [--compounds_modules_file_name COMPOUNS_MODULES_FILE_NAME] [--modules_genes_file_name MODULES_GENES_FILE_NAME]
               [--cycle_detection CYCLE_DETECTION] [--epoch_limit_all EPOCH_LIMIT_ALL] [--imbalance_loss_limit_all IMBALANCE_LOSS_LIMIT_ALL] [--save_predictions SAVE_PREDICTIONS] [--pca_components_selection PCA_COMPONENTS_SELECTION] [--n_epoch_scfea N_EPOCH_SCFEA]
               [--n_epoch_mpo N_EPOCH_MPO] [--n_epoch_snn N_EPOCH_SNN] [--do_train_snn DO_TRAIN_SNN] [--do_predict_snn DO_PREDICT_SNN] [--output_grad_snn OUTPUT_GRAD_SNN]

MPOSNN: A Massage Passing Optimizer-Based Supervised Neural Network Model to Estimate Cell-Wise Metabolic Using Single Cell RNA-seq Data.

options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        The inputs directory.
  --output_dir OUTPUT_DIR
                        The outputs directory, you can find all outputs in this directory.
  --geneExpression_file_name GENEEXPRESSION_FILE_NAME
                        The scRNA-seq file name.
  --compounds_modules_file_name COMPOUNS_MODULES_FILE_NAME
                        The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are
                        provided in /data/ folder.
  --modules_genes_file_name MODULES_GENES_FILE_NAME
                        The json file contains genes for each module. We provide human and mouse two models in scFEA. For human model, please use module_gene_m168.csv which is default. All candidate moduleGene files are provided in /data/ folder.
  --cycle_detection CYCLE_DETECTION
                        Remove the cycles in the graph. 0=False, 1=True
  --epoch_limit_all EPOCH_LIMIT_ALL
                        The user defined early stop Epoch(the whole framework)
  --imbalance_loss_limit_all IMBALANCE_LOSS_LIMIT_ALL
                        The user defined early stop imbalance loss.
  --save_predictions SAVE_PREDICTIONS
                        Save results. 0=False, 1=True
  --pca_components_selection PCA_COMPONENTS_SELECTION
                        Apply PCA to reduce the dimension of features. 0=False, 1=True
  --n_epoch_scfea N_EPOCH_SCFEA
                        User defined Epoch for scFEA training.
  --n_epoch_mpo N_EPOCH_MPO
                        User defined Epoch for Message Passing Optimizer.
  --n_epoch_snn N_EPOCH_SNN
                        User defined Epoch for Supervised Neural Network training.
  --do_train_snn DO_TRAIN_SNN
                        Train the SNN model, 0=False, 1=True.
  --do_predict_snn DO_PREDICT_SNN
                        Predict the flux values via the trained SNN model, 0=False, 1=True. FYI: If you have already trained the SNN model, SNN saves the model automatically, then you can set 
  --do_train_snn 0 and --do_predict_snn 1 to predict the flux valuesdirectly.
  --output_grad_snn OUTPUT_GRAD_SNN
                      Save the gradients on each gene.


```

## Inputs:

1. scRNA-seq data(rows:genes, columns:samples/cells)
2. Adjacency matrix a factor graph, you can choose one from "inputs"
   - 1:="The parent node(compound) of a reaction(M_i).", 
   - -1:="The child node(compound) of a reaction(M_i)", 
   - 0:="No connection bewteen compound and reaction". 
3. Modules(Reactions) and contained genes, you can choose one from "inputs".

## Outputs:

You can find the results in the directory "outputs", please click [here](https://github.com/ptdang1001/MPOSNN/tree/main/outputs) to see the output examples(Just for testing, no mearning):
1. "flux_scfea.csv", flux values from scFEA, rows:samples, columns:modules, each entry is a flux value.
2. "flux_snn.csv", flux values from SNN, rows:samples, columns:modules, each entry is a flux value.
3. "flux_snn_grad.csv", the gradients, rows:genes, columns:samples, each value represents the partial derivative of the model with respect to the gene.
4. "Compounds_Modules_FactorGraph_original.png", the visiulization of Factor Graph.
5. "flux_scFEA_MPO_SNN_std_scale_imbalance.png", the analysis of predicted flux values.  
    - 5.1. "module wise std":= $\frac{ \sum std(Y_{:,j}^{predicted})}{n}$
    - 5.2. "all mean scale":= $\frac{\sum \sum Y_{i,j}^{predicted}}{M*N}$
    - 5.3. "sample wise imbalance loss":= $\frac{\sum ImbalanceLoss(Y_{i,:})}{M}$


** Run code with default parameters: **
```
# just copy your data to the directory "inputs"
# The algorithm saves the model weights automatically
# you can get the results in the directory "outputs"

python src/main.py --geneExpression_file_name "your scRNA-seq data file name" --compounds_modules_file_name "your factor graph adj matrix file name" --modules_genes_file_name "your reactions and genes file name"

```

** Traning and Predicting example: **
```
# just copy your data to the directory "inputs"
# The algorithm saves the model weights automatically
# you can get the results in the directory "outputs"

python src/main.py --geneExpression_file_name "your scRNA-seq data file name" --compounds_modules_file_name "your factor graph adj matrix file name" --modules_genes_file_name "your reactions and genes file name" --do_train_snn 1 --do_predict_snn 1

```

** No Training, Predicting only example: **
```
# just copy your data to the directory "inputs"
# you can get the results in the directory "outputs"

python src/main.py --geneExpression_file_name "your scRNA-seq data file name" --compounds_modules_file_name "your factor graph adj matrix file name" --modules_genes_file_name "your reactions and genes file name" --do_train_snn 0 --do_predict_snn 1

```

## Questions & Problems

If you have any questions or problems, please feel free to open a new issue [here](https://github.com/ptdang1001/MPOSNN/issues). We will fix the new issue ASAP.  For code questions, please contact Pengtao Dang.

- [Pengtao Dang](https://ptdang1001.github.io/)
(pdang@iu.edu)

For any other further questions or requests, please contact the Principle Investigator of BDRL lab.
- [Prof. Chi Zhang](https://zcslab.github.io/)
(czhang87@iu.edu)

PhD candidate at [Biomedical Data Research Lab (BDRL)](https://zcslab.github.io/) , Indiana University School of Medicine


## Reference

1. N. Alghamdi, W. Chang, P. Dang, X. Lu, C. Wan, Z. Huang, J. Wang, M. Fishel, S. Cao, C. Zhang. scFEA: A graph neural network model to estimate cell-wise metabolic using single cell RNA-seq data, under review at *Genome Research*, 2020.
