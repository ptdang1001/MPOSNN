# -*- coding: utf8 -*

# built-in libs
import os
import sys
from functools import reduce
from collections import defaultdict,deque


# 3rd party libs
#import pysnooper
import numpy as np


#sys.path.append(os.path.abspath("./BP/utils"))


from MPO.utils.data_interface import build_bipartite_graph
from MPO.utils.data_interface import get_cycles


sep_sign = '*' * 100



# @pysnooper.snoop()
def pre_process_cycle(DG, nodes_in_cycle, nodes_in_cycle_set, factors_in_cycle, factors_in_cycle_set):
    factor_parentNode_childNode_outsideCycle = defaultdict(dict)
    startAnchorNode_factors_nextAnchorNode = defaultdict(dict)

    drop_nodes = []
    drop_factors = []
    cycle_valid_flag = False

    #################################################################################################################################
    # no anchor nodes in cycle
    if DG.nodes[nodes_in_cycle[0]]['n_parent']+DG.nodes[nodes_in_cycle[0]]['n_child'] <= 2:
        drop_nodes = nodes_in_cycle
        drop_factors = factors_in_cycle

        # get outside interaction factors in cycle
        outside_in_node = False
        outside_out_node = False
        for factor_i in factors_in_cycle:
            if DG.nodes[factor_i]['n_parent']+DG.nodes[factor_i]['n_child'] <= 2:
                continue

            outsideParentNode = list(
                filter(lambda x: (x not in nodes_in_cycle_set), list(DG.predecessors(factor_i)))) + []
            if len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[factor_i]['outsideParentNode'] = outsideParentNode

            outsideChildNode = list(filter(lambda x: (x not in nodes_in_cycle_set), list(DG.successors(factor_i)))) + []
            if len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[factor_i]['outsideChildNode'] = outsideChildNode

        if (outside_in_node and outside_out_node) or (not outside_in_node and not outside_out_node):
            cycle_valid_flag = True

        return cycle_valid_flag, drop_nodes, drop_factors, factor_parentNode_childNode_outsideCycle, startAnchorNode_factors_nextAnchorNode

    #################################################################################################################################
    # there is only one anchor node in cycle
    if DG.nodes[nodes_in_cycle[1]]['n_parent']+DG.nodes[nodes_in_cycle[1]]['n_child'] <= 2:

        start_anchor_node = nodes_in_cycle[0]

        # startAnchorNode_factors_nextAnchorNode[start_anchor_node]["next_drop_factors"] = []
        startAnchorNode_factors_nextAnchorNode[start_anchor_node]["next_anchor_node"] = start_anchor_node

        # get outside interaction factors in cycle
        outside_in_node = False
        outside_out_node = False
        for factor_i in factors_in_cycle:
            if DG.nodes[factor_i]['n_parent']+DG.nodes[factor_i]['n_child'] <= 2:
                continue

            outsideParentNode = list(
                filter(lambda x: (x not in nodes_in_cycle_set), list(DG.predecessors(factor_i)))) + []
            if not outside_in_node and len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[factor_i]['outsideParentNode'] = outsideParentNode
            outsideChildNode = list(filter(lambda x: (x not in nodes_in_cycle_set), list(DG.successors(factor_i)))) + []
            if not outside_out_node and len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[factor_i]['outsideChildNode'] = outsideChildNode

        startAnchorNode_factors_nextAnchorNode[start_anchor_node]["next_outIteraction_factors"] = list(factor_parentNode_childNode_outsideCycle.keys())


        if (outside_in_node and outside_out_node) or (not outside_in_node and not outside_out_node):
            cycle_valid_flag = True

        # if there are out outInteraction_factors
        if len(factor_parentNode_childNode_outsideCycle) > 0:
            drop_nodes = nodes_in_cycle[1:]
            drop_factors = factors_in_cycle

        return cycle_valid_flag, drop_nodes, drop_factors, factor_parentNode_childNode_outsideCycle, startAnchorNode_factors_nextAnchorNode

    #################################################################################################################################
    # there are at least 2 anchor nodes in the cycle

    outInteraction_factors = []
    next_drop_factors = []
    start_anchor_node = nodes_in_cycle.copy()[0]
    # startAnchorNode_factors_endAnchorNode[start_anchor_node]['next_factors'] = []
    # startAnchorNode_factors_endAnchorNode[start_anchor_node]['next_anchor_node'] =

    cur_anchor_node = nodes_in_cycle.copy()[0]
    # next_anchor_node=find_next_anchor_node(DG,cur_anchor_node,nodes_in_cycle_set,factors_in_cycle_set)
    next_factor = list(filter(lambda x: (x in factors_in_cycle_set), list(DG.successors(cur_anchor_node))))[0]
    next_node = list(filter(lambda x: (x in nodes_in_cycle_set), list(DG.successors(next_factor))))[0]

    while True:

        if DG.nodes[next_factor]['n_parent']+DG.nodes[next_factor]['n_child'] > 2:
            outInteraction_factors.append(next_factor)

            outsideParentNode = list(
                filter(lambda x: (x not in nodes_in_cycle_set), DG.predecessors(next_factor))) + []
            if len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[next_factor]['outsideParentNode'] = outsideParentNode

            outsideChildNode = list(filter(lambda x: (x not in nodes_in_cycle_set), DG.successors(next_factor))) + []
            if len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[next_factor]['outsideChildNode'] = outsideChildNode


        else:
            next_drop_factors.append(next_factor)
            drop_factors.append(next_factor)

        next_factor = list(filter(lambda x: (x in factors_in_cycle_set), DG.successors(next_node)))[0]

        if DG.nodes[next_node]['n_parent']+DG.nodes[next_node]['n_child'] > 2:
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                'next_outInteraction_factors'] = outInteraction_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_drop_factors'] = next_drop_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_anchor_node'] = next_node
            cur_anchor_node = next_node
            outInteraction_factors = []
            next_drop_factors = []
        else:
            drop_nodes.append(next_node)

        next_node = list(filter(lambda x: (x in nodes_in_cycle_set), DG.successors(next_factor)))[0]
        if next_node == start_anchor_node:
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                'next_outInteraction_factors'] = outInteraction_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_drop_factors'] = next_drop_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_anchor_node'] = start_anchor_node
            break

    if (not cycle_valid_flag) and (
            (outside_in_node and outside_out_node) or (not outside_in_node and not outside_out_node)):
        cycle_valid_flag = True

    return cycle_valid_flag, drop_nodes, drop_factors, factor_parentNode_childNode_outsideCycle, startAnchorNode_factors_nextAnchorNode


def find_start_anchor_node(startAnchorNode_factors_nextAnchorNode):
    for anchor_node_i in startAnchorNode_factors_nextAnchorNode.keys():
        next_outInteraction_factors = startAnchorNode_factors_nextAnchorNode[anchor_node_i][
            'next_outInteraction_factors']
        next_anchor_node = startAnchorNode_factors_nextAnchorNode[anchor_node_i]['next_anchor_node']

        if len(next_outInteraction_factors) == 0 and next_anchor_node:
            return next_anchor_node

    return list(startAnchorNode_factors_nextAnchorNode.keys())[0]



#@pysnooper.snoop()
def find_otherNodes_affectedBy_anchorNodes(DG, anchor_node, by="BFS"):
    q = deque()
    visited = set()
    q.append(anchor_node)
    # visited.add(anchor_node)
    all_affected_nodes = []

    while q:
        for _ in range(len(q)):
            cur = q.popleft()

            if cur not in visited and DG.nodes[cur]['type'] == "node":
                visited.add(cur)

                parent_child_factors = list(DG.predecessors(cur))+list(DG.successors(cur))
                parent_child_factors = list(filter(lambda x: (x not in visited), parent_child_factors))
                if len(parent_child_factors) == 0:
                    continue

                for factor_i in parent_child_factors:
                    visited.add(factor_i)

                    parent_child_nodes = list(DG.predecessors(factor_i))+list(DG.successors(factor_i))
                    if len(parent_child_nodes) > 2:
                        continue

                    affected_nodes = list(filter(lambda x: (x not in visited), parent_child_nodes))
                    all_affected_nodes.extend(affected_nodes)
                    q.extend(affected_nodes)
                    # visited=visited.union(set(affected_nodes))

                continue

            # if not visited(cur) and DG.nodes[cur]['type']=='factor':
            #    continue

    return all_affected_nodes





#@pysnooper.snoop()
def collapse_cycles(factors_nodes, DG, cycles_in_graph):
    # factors = set(factors_nodes.index.values)
    # nodes = set(factors_nodes.columns.values)

    # collapse the cycle
    all_drop_factors = set()
    all_drop_nodes = set()
    visited_outside_nodes = set()

    for i in range(len(cycles_in_graph)):
        factorsAndNodes_in_cycle = cycles_in_graph[i]
        nodes_in_cycle = list(filter(lambda x: DG.nodes[x]['type'] == 'node', factorsAndNodes_in_cycle))
        nodes_in_cycle.sort(key=lambda x: -(DG.nodes[x]['n_parent']+DG.nodes[x]['n_child']))
        nodes_in_cycle_set = set(nodes_in_cycle)

        factors_in_cycle = list(filter(lambda x: DG.nodes[x]['type'] == 'factor', factorsAndNodes_in_cycle))
        factors_in_cycle_set = set(factors_in_cycle)

        cycle_valid_flag, drop_nodes, drop_factors, factor_parentNode_childNode_outsideCycle, startAnchorNode_factors_nextAnchorNode = pre_process_cycle(
            DG, nodes_in_cycle, nodes_in_cycle_set, factors_in_cycle, factors_in_cycle_set)

        #################################################################################################################################
        # the cycle is valid
        if not cycle_valid_flag:
            print("\nThe cycle is not valid!!\n There is only In node or Out node linked to factors in the cycle!\n")
            return

        ##################################################################################################################################
        # no anchor node in the cycle
        # merger all anchor factors into one collapse_factor and link all in and out nodes to this collapse_factor
        # remove all nodes and factors in cycle
        if len(startAnchorNode_factors_nextAnchorNode) == 0:
            all_drop_factors = all_drop_factors.union(set(drop_factors))
            all_drop_nodes = all_drop_nodes.union(set(drop_nodes))

            collapse_factor = "collapse_factor_" + '_'.join(list(factor_parentNode_childNode_outsideCycle.keys()))
            factors_nodes.loc[collapse_factor, :] = 0
            for interaction_factor_i in factor_parentNode_childNode_outsideCycle.keys():
                parent_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i]['outsideParentNode']
                child_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i]['outsideChildNode']
                if parent_nodes:
                    for parent_node_i in parent_nodes:
                        if parent_node_i in visited_outside_nodes:
                            continue
                        factors_nodes.loc[collapse_factor, parent_node_i] = 1
                        visited_outside_nodes.add(parent_node_i)

                if child_nodes:
                    for child_node_i in child_nodes:
                        if child_node_i in visited_outside_nodes:
                            continue
                        factors_nodes.loc[collapse_factor, child_node_i] = -1
                        visited_outside_nodes.add(child_node_i)

            continue

        #################################################################################################################################
        # only one anchor node in cycle
        if len(startAnchorNode_factors_nextAnchorNode.keys()) == 1:
            start_anchor_node = None
            start_anchor_node = list(startAnchorNode_factors_nextAnchorNode.keys()).copy()[0]

            # no in node and no out node linked to the factors in the cycle
            if len(factor_parentNode_childNode_outsideCycle) == 0:
                print("There are no in and out interaction factors in the cycle!")
                # break the edge from the anchor node and its in cycle parent factor

                inCycle_parentFactor = \
                list(filter(lambda x: x in factors_in_cycle_set, list(DG.predecessors(start_anchor_node)))).copy()[0]
                factors_nodes.loc[inCycle_parentFactor, start_anchor_node] = 0




            else:
                all_drop_factors = all_drop_factors.union(set(drop_factors))
                all_drop_nodes = all_drop_nodes.union(set(drop_nodes))

                # There are in edge and out edge linked to the factors in the cycle
                # start_anchor_node = next(iter(startAnchorNode_factors_nextAnchorNode))

                collapse_factor = "collapse_factor_" + '_'.join(list(factor_parentNode_childNode_outsideCycle.keys()))
                factors_nodes.loc[collapse_factor, :] = 0
                for interaction_factor_i in factor_parentNode_childNode_outsideCycle.keys():
                    parent_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i]['outsideParentNode']
                    child_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i]['outsideChildNode']
                    if parent_nodes:
                        for parent_node_i in parent_nodes:
                            if parent_node_i in visited_outside_nodes:
                                continue
                            factors_nodes.loc[collapse_factor, parent_node_i] = 1
                            visited_outside_nodes.add(parent_node_i)

                    if child_nodes:
                        for child_node_i in child_nodes:
                            if child_node_i in visited_outside_nodes:
                                continue
                            factors_nodes.loc[collapse_factor, child_node_i] = -1
                            visited_outside_nodes.add(child_node_i)

                continue

        #################################################################################################################################
        # more than one anchor node in cycle
        if len(startAnchorNode_factors_nextAnchorNode.keys()) > 1:
            all_drop_nodes = all_drop_nodes.union(set(drop_nodes))
            all_drop_factors = all_drop_factors.union(set(drop_factors))

            ##################################################################################################################################
            # no in node and no out node linked to the factors in the cycle
            if len(factor_parentNode_childNode_outsideCycle) == 0:
                print("There are no in and out interaction factors in the cycle!")
                collapse_node = "collapse_node_" + '_'.join(
                    list(map(lambda x: x.split('_')[1], list(startAnchorNode_factors_nextAnchorNode.keys()))))
                factors_nodes.loc[:, collapse_node] = 0

                for anchor_node_i in startAnchorNode_factors_nextAnchorNode.keys():
                    all_drop_nodes.add(anchor_node_i)

                    parent_factor_outsideCycle = list(
                        filter(lambda x: (x not in factors_in_cycle_set), DG.predecessors(anchor_node_i))) + []
                    if parent_factor_outsideCycle:
                        for parent_factor_i in parent_factor_outsideCycle:
                            factors_nodes.loc[parent_factor_i, collapse_node] = -1

                    child_factor_outsideCycle = list(
                        filter(lambda x: (x not in factors_in_cycle_set), DG.successors(anchor_node_i))) + []
                    if child_factor_outsideCycle:
                        for child_factor_i in child_factor_outsideCycle:
                            factors_nodes.loc[child_factor_i, collapse_node] = 1

            else:
                #################################################################################################################################
                # at least one factor has in node and out node in the cycle
                start_anchor_node = find_start_anchor_node(startAnchorNode_factors_nextAnchorNode)
                collapse_factors = []
                collapse_factor_id = 1

                cur_anchor_node = start_anchor_node
                while True:
                    next_anchor_node = startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_anchor_node']
                    next_outInteraction_factors = startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                        'next_outInteraction_factors']
                    # next_drop_factors = startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_drop_factors']

                    # next anchor node is the start anchor node
                    if next_anchor_node == start_anchor_node:
                        # no interaction factors between the two anchor nodes
                        if len(next_outInteraction_factors) == 0:
                            break
                        else:
                            # there are out interaction factors between the two anchor nodes
                            collapse_factors.extend(next_outInteraction_factors)

                    else:
                        # no interaction factors between the two anchor nodes
                        if len(next_outInteraction_factors) == 0:
                            collapse_factor = "collapse_factor_" + str(collapse_factor_id).zfill(2)
                            collapse_factor_id += 1
                            factors_nodes.loc[collapse_factor, :] = 0
                            factors_nodes.loc[collapse_factor, cur_anchor_node] = 1
                            factors_nodes.loc[collapse_factor, next_anchor_node] = -1




                        else:
                            # there are out node interaction factors between the two anchor nodes
                            collapse_factors.extend(next_outInteraction_factors)

                            collapse_factor = "collapse_factor_" + '_'.join(next_outInteraction_factors)
                            factors_nodes.loc[collapse_factor, :] = 0
                            factors_nodes.loc[collapse_factor, cur_anchor_node] = 1
                            factors_nodes.loc[collapse_factor, next_anchor_node] = -1
                            for interaction_factor_i in next_outInteraction_factors:
                                all_drop_factors.add(interaction_factor_i)

                                parent_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i][
                                    'outsideParentNode']
                                child_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i][
                                    'outsideChildNode']
                                if parent_nodes:
                                    for parent_node_i in parent_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[collapse_factor, parent_node_i] = 1
                                        # visited_outside_nodes.add(parent_node_i)

                                if child_nodes:
                                    for child_node_i in child_nodes:
                                        # if child_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[collapse_factor, child_node_i] = -1
                                        # visited_outside_nodes.add(child_node_i)

                            # collapse all the interactive factors into one factor and link all outside nodes to this collapse_factor
                            collapse_factor = "collapse_factor_" + '_'.join(collapse_factors)

                            # the new collapse factor is already created then don't need to create it again
                            if collapse_factor in set(factors_nodes.index.values):
                                # collapse_factor="collapse_factor_"+'_'.join(collapse_factors)+'_'+str(collapse_factor_id).zfill(2)
                                # collapse_factor_id+=1
                                continue

                            factors_nodes.loc[collapse_factor, :] = 0
                            for interaction_factor_i in collapse_factors:
                                parent_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i][
                                    'outsideParentNode']
                                child_nodes = factor_parentNode_childNode_outsideCycle[interaction_factor_i][
                                    'outsideChildNode']
                                if parent_nodes:
                                    for parent_node_i in parent_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[collapse_factor, parent_node_i] = 1
                                        # visited_outside_nodes.add(parent_node_i)

                                if child_nodes:
                                    for child_node_i in child_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[collapse_factor, child_node_i] = -1
                                        # visited_outside_nodes.add(child_node_i)

    # remove all drop factors and nodes
    factors_nodes.drop(all_drop_nodes, axis=1, inplace=True)
    factors_nodes.drop(all_drop_factors, axis=0, inplace=True)

    return factors_nodes



def save_CycleCollapsed_factors_nodes(factors_nodes,BDG,cycles_in_graph,module_dir,module_source):
    has_cycle = False if len(cycles_in_graph)==0 else True

    if not has_cycle:
        print("\nNo Cycles in Graph!\n")
        return None,None

    print("\nRunning Cycle Collapsing...............................\n")

    factors_nodes_collapsedCycles = factors_nodes.copy()


    cycle_id = 1
    while len(cycles_in_graph) > 0:
        print(sep_sign)
        print("\n There are cycles in the graph\n Processing the cycles in the graph\n")
        print("Factors and Nodes in cycle:\n{0}\n".format(cycles_in_graph))

        factors_nodes_collapsedCycles = collapse_cycles(factors_nodes_collapsedCycles.copy(), BDG,cycles_in_graph)
        factors_nodes_collapsedCycles = factors_nodes_collapsedCycles.astype(int)
        print(factors_nodes_collapsedCycles)
        print("\nCycle {0} collapsed!\n".format(cycle_id))
        cycle_id += 1

        BDG = None
        cycles_in_graph = None
        BDG= build_bipartite_graph(factors_nodes_collapsedCycles)
        cycles_in_graph=get_cycles(BDG)

    factors_nodes_collapsedCycles.to_csv(module_dir + module_source + "_cmMat_collapsedCycles.csv")
    print("\n New Cycles Collapsed factors_nodes file saved!!\n")
    return BDG,factors_nodes_collapsedCycles




class MPO:
    def __init__(self, belief, factors, nodes, anchor_nodes ,args):
        self.factors = factors
        self.nodes = nodes
        self.nodes_in_cycle=[]
        self.args=args

        self.belief = abs(belief)
        self._msg_factors2nodes = {}
        self._msg_nodes2factors = {}
        self._belief_new_set = []

        self._init_msg_factors2nodes()
        self._init_msg_nodes2factors()

    def get_belief_diff_factor2node(self, cur_node, parent_nodes, child_nodes):
        # other_neighbor_nodes = parent_nodes.union(child_nodes)
        # other_neighbor_nodes.remove(cur_node)
        # other_neighbor_nodes_len = len(other_neighbor_nodes)

        belief_sum_other_parent_nodes = 0.0
        for parent_node_i in parent_nodes:
            if parent_node_i == cur_node:
                continue
            belief_sum_other_parent_nodes += float(self.belief[parent_node_i])
        # print("msg_other_neighbors2parent_nodes:\n",msg_other_neighbors2parent_nodes)

        belief_sum_other_child_nodes = 0.0
        for child_node_i in child_nodes:
            if child_node_i == cur_node:
                continue

            belief_sum_other_child_nodes += float(self.belief[child_node_i])
        # print("msg_other_neighbors2child_nodes:\n",msg_other_neighbors2child_nodes)

        belief_difference = abs(belief_sum_other_parent_nodes - belief_sum_other_child_nodes)

        # msg_factor2node = (msg_other_neighbors2parent_nodes + msg_other_neighbors2child_nodes)/2
        # self._msg_factors2nodes[(factor, cur_node)] = msg_factor2node
        return belief_difference

    # @pysnooper.snoop()
    def _init_msg_factors2nodes(self):
        for factor_i, parent_child in self.factors.items():
            parent_nodes = parent_child["parent_nodes"]
            child_nodes = parent_child["child_nodes"]
            if len(parent_nodes) == 0 or (
                    len(parent_nodes) == 1 and list(parent_nodes)[0] == "dummy_parent_node_4_" + factor_i):

                for child_node_i in child_nodes:
                    msg = 0.0
                    child_node_i_parent_factor_len = len(self.nodes[child_node_i]["parent_factors"])
                    if child_node_i_parent_factor_len == 1:
                        msg = float(self.belief[child_node_i])
                        self._msg_nodes2factors[(child_node_i, factor_i)] = msg
                    elif child_node_i_parent_factor_len > 1:
                        msg = float(self.belief[child_node_i])
                        self._msg_nodes2factors[(child_node_i, factor_i)] = msg


                    self._msg_nodes2factors[(list(parent_nodes)[0],factor_i)]=msg
                    self._msg_factors2nodes[(factor_i, child_node_i)] = msg
            else:
                for parent_node_i in parent_nodes:
                    belief_difference = self.get_belief_diff_factor2node(parent_node_i, parent_nodes, child_nodes)
                    self._msg_factors2nodes[(factor_i, parent_node_i)] = belief_difference

                for child_node_i in child_nodes:
                    belief_difference = self.get_belief_diff_factor2node(child_node_i, parent_nodes, child_nodes)
                    self._msg_factors2nodes[(factor_i, child_node_i)] = belief_difference

    # @pysnooper.snoop()
    def _init_msg_nodes2factors(self):
        for node_i, parent_child in self.nodes.items():
            parent_factors = parent_child["parent_factors"]
            child_factors = parent_child["child_factors"]
            all_neighbor_factors = parent_factors + child_factors

            dummy_factors = [factor_i for factor_i in all_neighbor_factors if factor_i.startswith("dummy")]
            for dummy_factor_i in dummy_factors:
                belief = float(self.belief[node_i])
                self._msg_nodes2factors[(node_i, dummy_factor_i)] = belief
                self._msg_factors2nodes[(dummy_factor_i, node_i)] = belief

            for parent_factor_i in parent_factors:
                if (node_i, parent_factor_i) in self._msg_nodes2factors or parent_factor_i.startswith("dummy"):
                    continue

                self.update_msg_node2factor(node_i, parent_factor_i, all_neighbor_factors)

            for child_factor_i in child_factors:
                if (node_i, child_factor_i) in self._msg_nodes2factors or child_factor_i.startswith("dummy"):
                    continue

                self.update_msg_node2factor(node_i, child_factor_i, all_neighbor_factors)

    # @pysnooper.snoop()
    def get_factor_weight(self, node, cur_factor, res_by="mean"):
        return 1

        other_neighbor_nodes = self.factors[cur_factor]["parent_nodes"] + self.factors[cur_factor]["child_nodes"]
        other_neighbor_nodes.remove(node)

        global get_factor_loss
        # @pysnooper.snoop()
        def get_factor_loss(factor):
            parent_nodes_belief_sum = float(sum(self.belief[self.factors[factor]["parent_nodes"]].values[0]))
            child_nodes_belief_sum = float(sum(self.belief[self.factors[factor]["child_nodes"]].values[0]))
            loss = (parent_nodes_belief_sum - child_nodes_belief_sum) ** 2
            return loss

        other_neighbor_nodes_factors_loss = []
        for node_i in other_neighbor_nodes:
            other_neighbor_factors = self.nodes[node_i]["parent_factors"] + self.nodes[node_i]["child_factors"]
            other_neighbor_factors.remove(cur_factor)
            other_neighbor_factors = list(
                filter(lambda factor_i: (not factor_i.startswith("dummy")), other_neighbor_factors))

            if len(other_neighbor_factors) == 0:
                continue

            loss = list(map(get_factor_loss, other_neighbor_factors))
            other_neighbor_nodes_factors_loss += loss
        # print("The loss of the neighbor factors:{0}\n".format(other_neighbor_nodes_factors_loss))
        total_loss = 1.0
        if len(other_neighbor_nodes_factors_loss) == 0:
            return total_loss

        if res_by == "mean":
            total_loss = np.mean(other_neighbor_nodes_factors_loss)
        elif res_by == "sum":
            total_loss = np.sum(other_neighbor_nodes_factors_loss)
        #total_loss = (1.0 / (1.0 + np.exp(-total_loss)))+1.0
        total_loss+=1.0

        return total_loss


    def get_n_node_parent(self,node_i):
        parent_factors=self.nodes[node_i]["parent_factors"]
        if len(parent_factors)==1 and parent_factors[0].startswith("dummy"):
            return 0
        else:
            return 1

    def get_n_node_child(self,node_i):
        child_factors=self.nodes[node_i]["child_factors"]
        if len(child_factors)==1 and child_factors[0].startswith("dummy"):
            return 0
        else:
            return 1


    # @pysnooper.snoop()
    def update_msg_factor2node(self, factor, cur_node, parent_nodes, child_nodes):
        if factor.startswith("dummy"):
            belief = float(self.belief[cur_node])
            self._msg_factors2nodes[(factor, cur_node)] = belief
            return True

        n_parent_node_no_parent=0
        msg_from_parent_nodes = 0.0
        for parent_node_i in parent_nodes:
            if parent_node_i == cur_node:
                continue

            if self.get_n_node_parent(parent_node_i)==0:
                n_parent_node_no_parent+=1

            msg_from_parent_nodes += self._msg_nodes2factors[(parent_node_i, factor)]
        # print("msg_other_neighbors2parent_nodes:\n",msg_other_neighbors2parent_nodes)


        n_child_node_no_child=0
        msg_from_child_nodes = 0.0
        for child_node_i in child_nodes:
            if child_node_i == cur_node:
                continue

            if self.get_n_node_child(child_node_i)==0:
                n_child_node_no_child+=1
            # tmp=self._msg_nodes2factors[(child_node_i,factor)]
            msg_from_child_nodes += self._msg_nodes2factors[(child_node_i, factor)]
        # print("msg_other_neighbors2child_nodes:\n",msg_other_neighbors2child_nodes)

        msg_factor2node = abs(msg_from_parent_nodes - msg_from_child_nodes)
        #if (n_parent_node_no_parent+n_child_node_no_child)==(len(parent_nodes)+len(child_nodes)-1):
        #    msg_factor2node=float(self.belief[cur_node])
        # print("msg_factor2node:",msg_factor2node)
        # sys.exit(1)
        #rdm_num=np.random.randn()/1000
        rdm_num=0

        self._msg_factors2nodes[(factor, cur_node)] = abs(msg_factor2node+rdm_num)

    # @pysnooper.snoop()
    def update_msg_factor2nodes(self, factor, parent_nodes, child_nodes):
        # update the msg(1 facor->other nodes(the factor's parent and child))
        for parent_node_i in parent_nodes:
            self.update_msg_factor2node(factor, parent_node_i, parent_nodes, child_nodes)

        for child_node_i in child_nodes:
            self.update_msg_factor2node(factor, child_node_i, parent_nodes, child_nodes)

    def update_msg_node2factor(self, node, cur_factor, all_neighbor_factors):
        other_neighbor_factors = all_neighbor_factors.copy()
        other_neighbor_factors.remove(cur_factor)
        #if len(other_neighbor_factors)==1 and other_neighbor_factors[0].startswith('dummy'):
        #    other_neighbor_factors=[]

        other_neighbor_factors_len = len(other_neighbor_factors)
        msg_node2factor = 0.0
        if other_neighbor_factors_len != 0:
            msg_node2factor = reduce(lambda x, y: x + y, list(map(lambda factor_i: (
                    self.get_factor_weight(node, factor_i) * self._msg_factors2nodes[(factor_i, node)]),
                                                                  other_neighbor_factors)) + [
                                         0]) / other_neighbor_factors_len
        else:
            msg_node2factor = float(self.belief[node])
            #msg_node2factor = self._msg_factors2nodes[(cur_factor,node)]

        self._msg_nodes2factors[(node, cur_factor)] = msg_node2factor

    def update_msg_node2factors(self, node, parent_factors, child_factors):
        # update the msg(1 node->other factors(the ndoe's parent and child))
        all_neighbor_factors = parent_factors + child_factors
        for factor_i in all_neighbor_factors:
            if factor_i.startswith("dummy"):
                self._msg_nodes2factors[(node, factor_i)] = float(self.belief[node])
                continue

            self.update_msg_node2factor(node, factor_i, all_neighbor_factors)



    # update_belief
    def update_belief(self):
        for node_i in self.belief.columns.values:
            all_neighbor_factors = self.nodes[node_i]["parent_factors"] + self.nodes[node_i]["child_factors"]
            #all_neighbor_factors = list(filter(lambda x: (not x.startswith('dummy_')), all_neighbor_factors))

            belief_new = reduce(lambda x, y: x + y, list(map(lambda factor_i: (
                    self.get_factor_weight(node_i, factor_i) * self._msg_factors2nodes[(factor_i, node_i)]),
                                                             all_neighbor_factors)) + [
                                    0]) / len(all_neighbor_factors)

            self.belief[node_i] = belief_new

            #for factor_i in all_neighbor_factors:
            #    self._msg_nodes2factors[(node_i, factor_i)] = belief_new
        # self.belief = self.belief.div(np.sum(self.belief.values))

        '''
        if len(self.nodes_in_cycle)==0:
            return True

        for nodes_1cycle in self.nodes_in_cycle:
            # print(self.belief.loc[:,nodes_1cycle])
            belief_avg = 0
            belief_avg = self.belief.loc[:, nodes_1cycle].mean(axis=1)
            self.belief.loc[:, nodes_1cycle] = belief_avg
            # print(self.belief.loc[:,nodes_1cycle])
            # print(belief_avg)
        '''


        return True

    # @pysnooper.snoop()
    def run(self):
        sep_sign = "*" * 100

        '''
        print(sep_sign)
        print("Factors:\n")
        print(self.factors)
        print("nodes:\n")
        print(self.nodes)

        print(sep_sign)
        print("init msg factors->nodes:\n{0}".format(self._msg_factors2nodes))
        print(sep_sign)
        print("init msg nodes->factors:\n{0}".format(self._msg_nodes2factors))
        #sys.exit(1)
        '''

        # update the msg_factors2nodes and msg_nodes2factors
        for i in range(1, (self.args.n_epoch_mpo + 1)):
            # print(sep_sign)
            # print("Epoch: {0}".format(i))

            # update the msg factors to nodes
            for factor_i, parent_child in self.factors.items():  # update msg_factors2nodes
                # for factor_i in sorted_factors:
                # print(sep_sign)
                # print("updating factor:{0}".format(factor_i))
                # parent_child=self.factors[factor_i]
                parent_nodes = parent_child["parent_nodes"]  # parent and child are nodes=flux values
                child_nodes = parent_child["child_nodes"]

                self.update_msg_factor2nodes(factor_i, parent_nodes, child_nodes)

            # print(sep_sign)
            # print("new msg factors->nodes:\n{0}".format(self._msg_factors2nodes))

            for node_i, parent_child in self.nodes.items():  # update msg_nodes2factors
                # for node_i in sorted_nodes:
                # print(sep_sign)
                # print("updating node:{0}".format(node_i))
                # parent_child=self.nodes[node_i]
                parent_factors = parent_child["parent_factors"]
                child_factors = parent_child["child_factors"]
                self.update_msg_node2factors(node_i, parent_factors, child_factors)

            # print(sep_sign)
            # print("new msg nodes->factors:\n{0}".format(self._msg_nodes2factors))

            # update  nodes' belief
            self.update_belief()
            # print(sep_sign)
            # print("new belief:\n{0}".format(self.belief))
            # print("new belief sum:{0}".format(np.sum(self.belief.values)))

            self._belief_new_set.append(self.belief.values.copy())

        return True
