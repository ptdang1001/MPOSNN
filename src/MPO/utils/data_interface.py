# -*- coding: utf8 -*


#build-in libs
import os
from multiprocessing import Pool
from collections import defaultdict

#3rd party libs
import numpy as np
import pandas as pd
import networkx as nx
#from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
#import pysnooper




def build_bipartite_graph(factors_nodes):
    # generate graph
    BDG = nx.DiGraph()

    node_names=factors_nodes.columns.values
    factor_names=factors_nodes.index.values
    # add node to graph
    for node_name in node_names:
        # get parent factors
        idx = None
        idx = np.where(factors_nodes.loc[:, node_name] == -1)[0]
        parent_factors = list(np.take(factor_names,
                                          idx)) if len(idx) else []
        # get child factors
        idx = None
        idx = np.where(factors_nodes.loc[:, node_name] == 1)[0]
        child_factors = list(np.take(factor_names,
                                         idx)) if len(idx) else []
        BDG.add_node(node_name, bipartite=1, type='node', color="red", shape='o',
                        n_parent=len(parent_factors), n_child=len(child_factors))

        if len(parent_factors)==0:
            dummy_factor="dummy_parent_4_"+node_name
            BDG.add_node(dummy_factor,bipartite=0,color="blue", shape='s',n_parent=0,n_child=1)
            BDG.add_edge(dummy_factor,node_name)
        if len(child_factors)==0:
            dummy_factor="dummy_child_4_"+node_name
            BDG.add_node(dummy_factor,bipartite=0,color="blue", shape='s',n_parent=1,n_child=0)
            BDG.add_edge(node_name,dummy_factor)

    for factor_name in factor_names:
        # get parent nodes
        idx = None
        idx = np.where(factors_nodes.loc[factor_name, :] == 1)[0]
        parent_nodes = list(np.take(node_names,
                                    idx)) if len(idx) else []

        # get child nodes
        idx = None
        idx = np.where(factors_nodes.loc[factor_name, :] == -1)[0]
        child_nodes = list(np.take(node_names,
                                       idx)) if len(idx) else []
        # add factor to graph
        BDG.add_node(factor_name, bipartite=0, type='factor', color="blue", shape='s',
                    n_parent=len(parent_nodes), n_child=len(child_nodes))

        # add edge from parent node to factor
        for parent_node_i in parent_nodes:
            BDG.add_edge(parent_node_i, factor_name)

        for child_node_i in child_nodes:
            BDG.add_edge(factor_name, child_node_i)  # generate graph

    factors=[factor_i for factor_i in BDG.nodes() if BDG.nodes[factor_i]['bipartite']==0]
    nodes=[node_i for node_i in BDG.nodes() if BDG.nodes[node_i]['bipartite']==1]


    return BDG,factors,nodes



class Factor_Graph:
    def __init__(self, factors_nodes):
        self.factors_nodes = factors_nodes
        self._factor_names = factors_nodes.index.values
        self._node_names = factors_nodes.columns.values
        self._factors = {}
        self._nodes = {}

        self._init_factors()
        self._init_nodes()


    def init_1_factor(self,factor):
            idx = np.where(self.factors_nodes.loc[factor, :] == 1)[0]
            parent = list(np.take(self._node_names, idx)) if len(idx) else []
            idx = np.where(self.factors_nodes.loc[factor, :] == -1)[0]
            child = list(np.take(self._node_names, idx)) if len(idx) else []
            #if len(parent) == 0:
            #    dummy = "dummy_parent_node_4_" + factor
            #    parent = [dummy]
            #if len(child) == 0:
            #    dummy = "dummy_child_node_4_" + factor
            #    child = [dummy]
            return factor, {"parent_nodes": parent, "child_nodes": child}


    # @pysnooper.snoop()
    def _init_factors(self):
        res = []
        with Pool(os.cpu_count()) as p:
            res.append(p.map(self.init_1_factor, self._factor_names))

        # for factor,parent_child in res:
        for factor, parent_child_node in res[0]:
            self._factors[factor] = parent_child_node


    def init_1_node(self,node):
            idx = np.where(self.factors_nodes[node] == -1)[0]
            parent = list(np.take(self._factor_names, idx)) if len(idx) else []
            idx = np.where(self.factors_nodes[node] == 1)[0]
            child = list(np.take(self._factor_names, idx)) if len(idx) else []
            if len(parent) == 0:
                dummy = "dummy_parent_factor_4_" + node
                parent = [dummy]
            if len(child) == 0:
                dummy = "dummy_child_factor_4_" + node
                child = [dummy]
            return node, {"parent_factors": parent, "child_factors": child}


    # @pysnooper.snoop()
    def _init_nodes(self):
        res = []
        with Pool(os.cpu_count()) as p:
            res.append(p.map(self.init_1_node, self._node_names))

        for node, parent_child_factor in res[0]:
            self._nodes[node] = parent_child_factor




def remove_allZero_rowAndCol(factors_nodes):
    # remove all zero rows and columns
    factors_nodes = factors_nodes.loc[~(factors_nodes == 0).all(axis=1), :]
    factors_nodes = factors_nodes.loc[:, ~(factors_nodes == 0).all(axis=0)]
    return factors_nodes



def plot_graph(factors_nodes, file_dir, file_name):
    # generate graph
    DG = nx.DiGraph()

    # add node to graph
    for node_name in factors_nodes.columns.values:
        DG.add_node(node_name, bipartite=1,
                    color="red",
                    shape='o')

    for factor_name in factors_nodes.index.values:
        # add factor to graph
        DG.add_node(factor_name, bipartite=1, color="blue", shape='s')

        # get parent nodes
        idx = np.where(factors_nodes.loc[factor_name, :] == 1)[0]
        parent_nodes = list(np.take(factors_nodes.columns.values, idx)) if len(idx) else []

        # get child nodes
        idx = np.where(factors_nodes.loc[factor_name, :] == -1)[0]
        child_nodes = list(np.take(factors_nodes.columns.values, idx)) if len(idx) else []

        # add edge from parent node to factor
        for parent_node_i in parent_nodes:
            DG.add_edge(parent_node_i, factor_name)

        for child_node_i in child_nodes:
            DG.add_edge(factor_name, child_node_i)

    colors = [DG.nodes[i]['color'] for i in DG.nodes()]
    shapes = [DG.nodes[i]['shape'] for i in DG.nodes()]

    cycles_in_graph = list(nx.simple_cycles(DG))

    # plt.figure()
    figure(figsize=(18, 18))

    pos = nx.nx_pydot.graphviz_layout(DG)
    # pos = nx.spring_layout(DG)
    # pos = nx.kamada_kawai_layout(DG)
    # pos=nx.circular_layout(DG)
    # pos = nx.shell_layout(DG)
    # pos = nx.spectral_layout(DG)
    # pos=nx.random_layout(DG)

    # node_size = 200

    nx.draw(DG, pos, with_labels=True)
    nx.draw_networkx_nodes(DG,
                           pos,
                           nodelist=factors_nodes.columns.values,
                           node_color="yellow",
                           node_shape='o',
                           # node_size=node_size
                           )
    nx.draw_networkx_nodes(DG,
                           pos,
                           nodelist=factors_nodes.index.values,
                           node_color="lightblue",
                           node_shape='s',
                           # node_size=node_size
                           )
    nx.draw_networkx_edges(DG,
                           pos,
                           width=1.5,
                           arrowsize=20
                           )

    if len(cycles_in_graph) > 0:
        for cycle in cycles_in_graph:
            nx.draw_networkx_nodes(DG,
                                   pos,
                                   nodelist=cycle,
                                   node_color="red",
                                   node_shape='o',

                                   )

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    # plt.show()
    plt.savefig(file_dir + file_name)

    return DG


def get_cycles(BDG):
    cycles_in_graph = list(nx.simple_cycles(BDG))
    cycles_in_graph.sort(key=lambda x: -len(x))
    return cycles_in_graph



#@pysnooper.snoop()
def get_anchor_nodes(factors_nodes, DG):
    anchorNodes_affectedNodes = defaultdict(list)
    for node_i in factors_nodes.columns.values:
        parent_factors = list(DG.predecessors(node_i))
        child_factors = list(DG.successors(node_i))


        if parent_factors and child_factors and len(parent_factors) + len(child_factors) >= 3:
            affected_nodes = find_otherNodes_affectedBy_anchorNodes(DG, node_i)
            anchorNodes_affectedNodes[node_i] = affected_nodes

    return anchorNodes_affectedNodes


def get_cosine_similarity(belief_1, belief_prediction_lists):
    cosine_similarity = []
    for belief_new in belief_prediction_lists:
        tmp = np.dot(belief_1.values, belief_new) / (np.linalg.norm(belief_1.values) * np.linalg.norm(belief_new))
        cosine_similarity.append(tmp)
    return cosine_similarity


#@pysnooper.snoop()
def get_imbalanceLoss(factors_nodes, belief_set):
    imbalanceLoss_values = []
    for belief_i in belief_set:
        tmp = 0
        max_val=max(belief_i)
        min_val=min(belief_i)
        dif=max_val-min_val if max_val>min_val else 1
        if dif<=0.001:
            dif=1
        belief_new = np.true_divide(belief_i, dif)
        #belief_new=np.true_divide(belief_i,np.linalg.norm(belief_i))
        tmp1=belief_new * factors_nodes.values
        tmp2=np.sum(tmp1, axis=1)
        tmp3=tmp2**2
        tmp4=np.sum(tmp3)
        tmp = np.round(tmp4, 4)

        imbalanceLoss_values.append(tmp4)

    return imbalanceLoss_values


