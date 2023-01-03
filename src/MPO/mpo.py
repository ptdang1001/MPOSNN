# -*-coding:utf-8-*-

import os
import sys
#import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
#import pysnooper

# my libs



from MPO.utils.data_interface import Factor_Graph
from MPO.utils.data_interface import get_imbalanceLoss
from MPO.utils.model_interface import MPO

sep_sign = '*' * 100


def get_one_sample_flux(sample_name, factors_nodes, belief_old, factors, nodes, args):
    mpo = MPO(belief_old.copy(), factors, nodes, [], args)

    # run the belief propagation
    mpo.run()

    belief_predicted_set = []
    for belief in mpo._belief_new_set:
        belief_predicted_set.append(belief[0])

    belief_predicted_set = np.stack(belief_predicted_set)
    # print(belief_predicted_set)

    imbalanceLoss_values = get_imbalanceLoss(factors_nodes, belief_predicted_set)
    # print("\nall imbalanceLoss_values:{0}\n".format(imbalanceLoss_values))
    min_idx = imbalanceLoss_values.index(min(imbalanceLoss_values))
    belief_predicted = belief_predicted_set[min_idx]
    return sample_name, belief_predicted


# @pysnooper.snoop()
def run_mpo(factors_nodes, samples_modules_input, args):
    factor_graph = Factor_Graph(factors_nodes)  # This is a bipartite graph.
    samples_modules_mpo = {}

    res = []
    pool = Pool(os.cpu_count())
    for sample_i in range(samples_modules_input.shape[0]):
        sample_name=samples_modules_input.index.values[sample_i]
        belief_old = None
        belief_old = samples_modules_input.iloc[sample_i, :].values.tolist()
        belief_old = pd.DataFrame(belief_old)
        belief_old = belief_old.T
        belief_old.columns = factors_nodes.columns
        res.append(pool.apply_async(func=get_one_sample_flux,
                                    args=(
                                    sample_name, factors_nodes, belief_old, factor_graph._factors, factor_graph._nodes,
                                    args)))
    for res_i in res:
        sample_name,belief_predicted=res_i.get()
        samples_modules_mpo[sample_name]=belief_predicted

    samples_modules_mpo=pd.DataFrame.from_dict(samples_modules_mpo,orient='index')
    samples_modules_mpo.columns=samples_modules_input.columns
    samples_modules_mpo.index=samples_modules_input.index
    return samples_modules_mpo


# @pysnooper.snoop()
def mpo(compounds_modules, samples_modules_input, args):
    samples_modules_mpo = None
    samples_modules_mpo = run_mpo(compounds_modules, samples_modules_input, args)
    samples_modules_mpo = samples_modules_mpo.abs()
    return samples_modules_mpo
