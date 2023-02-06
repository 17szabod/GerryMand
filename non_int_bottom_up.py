# Core idea: Enumerate Motzkin paths into preallocated data structure
# Each contiguous section of the frontier has its own motzkin paths
# The 1 could be anywhere in the start_edge's contiguous section
# Maintain a dictionary of edge relations
# Motzkin paths should be enumerated by putting in either a 0 or a 3 or 2 if 3 is before
# List of dicts?
# Only main algorithm is bottom up, rest can be top down

import collections
import copy
import functools
import gc
import itertools
import math
import os
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

import generate_face_order
import poly_point_isect
import sqlite3

debug = False
depth_bound = 4


# https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partition(number, p_count):
    answer = set()
    if p_count == 1:
        answer.add((number,))
        return answer
    for x in range(0, number + 1):
        for y in partition(number - x, p_count - 1):
            answer.add((x,) + y)
    return answer


def count_non_int_paths_w_table(table, edge_dicts, k, num_samples, conn=None, just_sample=False):
    num_states = 2*(k-1) + 1
    # Sample top down:
    # sample paths are of the form (0:path, 1:end_state)
    sample_paths = [{0: [''], 1: 0} for x in range(num_samples)]

    if conn is not None:
        cur = conn.cursor()
        # count
        if not just_sample:
            # base count
            cur.execute("UPDATE nodes_{0} set _count=1 WHERE node_name='{1}';".format(len(table)-1, '.' + str(num_states-1)))
            conn.commit()
            for i in range(len(table)-1):
                ind = len(table) - i - 1
                # update counts
                sql_add = """update {0} set _count = tmp._count
                     from (select {0}.node_name, coalesce(sum({2}._count), 0) as _count
                           from (({0} left join {1} on {0}.node_name = {1}.end_node)
                               left join {2} on ({2}.node_name = {1}.start_node))
                           group by {0}.node_name) as tmp
                     where {0}.node_name=tmp.node_name;""".format("nodes_" + str(ind-1), "map_" + str(ind-1), "nodes_" + str(ind))
                cur.execute(sql_add)
                conn.commit()
                # map_query = "SELECT end_node FROM {0} WHERE start_node=?".format("map_" + str(ind))
                # node_query = "SELECT * FROM {0} WHERE node_name IN ".format("nodes_" + str(i))
                # update_query = "UPDATE {0} SET _count=? WHERE node_name=?".format("nodes_" + str(i+1))
                # for s in range(num_states):
                #     cur.execute("SELECT * FROM {0}".format("nodes_" + str(ind)))
                #     nodes = cur.fetchall()
                #     for node, c in nodes:
                #         cur.execute(map_query, (node,))
                #         rows = cur.fetchall()
                #         for r in rows:
                #             cur.execute("SELECT ")
                #         update_query = "UPDATE {0} SET _count=? WHERE node_name=?".format("nodes_" + str(i))
                #         cur.execute(update_query, (sum(count_arr), sample_paths[-1]))
                #
                #     for path in edge_dicts[ind][s]:
                #         for tup in edge_dicts[ind][s][path]:
                #             if tup[0] in table[ind][tup[1]] and table[ind+1][s][path] > 0:
                #                 table[ind][tup[1]][tup[0]] += table[ind+1][s][path]
        # sample
        for i in range(len(table)-1):  # go from 0 to the last map
            map_query = "SELECT start_node FROM {0} WHERE end_node=?".format("map_" + str(i))
            node_query = "SELECT * FROM {0} WHERE node_name IN ".format("nodes_" + str(i+1))
            for j in range(len(sample_paths)):
                path = sample_paths[j][0]
                s = sample_paths[j][1]
                cur.execute(map_query, (path[-1] + "." + str(s),))
                rows = cur.fetchall()
                cur.execute(node_query + "({0})".format(', '.join(['?']*len(rows))), tuple(r[0] for r in rows))
                relevant_nodes = cur.fetchall()
                count_arr = [x[1] for x in relevant_nodes]
                if len(count_arr) == 0:
                    print("Lost a path :(")  # Really should never happen, unless layer has an empty entry
                choice = np.random.uniform(0, sum(count_arr))
                sample_ind = np.arange(len(count_arr))[
                    np.asanyarray([sum(count_arr[:x + 1]) >= choice for x in range(len(count_arr))])][0]
                # Set new path_k and append next step in path
                name, state = relevant_nodes[sample_ind][0].split(".")
                path.append(name)
                sample_paths[j][1] = state
    else:
        rev_edge_maps = []  # Keep track of backtracking edge maps for sampling
        # table[-1][num_states-1][''] = 1
        for i in range(1, len(table)):  # Go through bottom up (always subtract i)
            edge_map = [collections.defaultdict(list) for k in range(num_states)]
            ind = len(table) - i - 1
            for s in range(num_states):
                for path in edge_dicts[ind][s]:
                    for tup in edge_dicts[ind][s][path]:
                        if tup[0] in table[ind][tup[1]] and table[ind+1][s][path] > 0:
                            table[ind][tup[1]][tup[0]] += table[ind+1][s][path]
                            edge_map[tup[1]][tup[0]].append((path, s, table[ind+1][s][path]))
            rev_edge_maps.append(edge_map)
        rev_edge_maps = list(reversed(rev_edge_maps))
        # for y in range(num_states - 1):  # Allstates except the first one
        #     sample_paths.append([])  # for s=1,...2(k-1), the init path has no districts
        for i in range(len(rev_edge_maps)):
            for j in range(len(sample_paths)):
                path = sample_paths[j][0]
                path_k = sample_paths[j][1]
                count_arr = [x[2] for x in rev_edge_maps[i][path_k][path[-1]]]
                if len(count_arr) == 0:
                    print("Lost a path :(")  # Really should never happen, unless layer has an empty entry
                choice = np.random.uniform(0, sum(count_arr))
                sample_ind = np.arange(len(count_arr))[
                    np.asanyarray([sum(count_arr[:x + 1]) >= choice for x in range(len(count_arr))])][0]
                # Set new path_k and append next step in path
                new_step, new_k, c = rev_edge_maps[i][path_k][path[-1]][sample_ind]
                sample_paths[j][0].append(new_step)
                sample_paths[j][1] = new_k
            # print("Currently have {0} paths".format(len(sample_paths)))
    # print(sample_paths)
    return sample_paths, table[0][0][''] if conn is None else 0


# @profile
def count_non_int_paths(face_list, outer_boundary, cont_sections, k):
    # Loop through reversed cont_sections - keep track of cur_sect and prev_sect
    # Generate and loop through each motzkin path of cur_sect and find connected path in prev_sect
    # Add the values of each of cur_sect path's neighbors from prev_sect to its value
    # Store these in prev_dict and cur_dict
    # prev_dict and cur_dict should have an entry for each state! Complexity is O(2^{2(k-1)\sqrt{n}})
    # prev_dict = cur_dict
    init_path = ''
    num_states = 2*(k-1) + 1  # The number of possible states the boundary can be at
    # prev_dict = [{init_path: 0} for x in range(num_states)]  # Do we need to store which step it is in these dicts? No!
    prev_dict = [collections.defaultdict() for x in range(num_states)]  # Do we need to store which step it is in these dicts? No!
    prev_dict[0][init_path] = 1
    # sample_tree holds, for each state, a tree of possible paths starting from the initial layer
    sample_tree = [[{}] for x in range(num_states)]
    sample_tree[0][0][init_path] = [1, []]  # in the first layer, only s=0 has anything
    prev_sect = cont_sections[-1]
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    num_samples = 1000
    overhead = 8
    subtree_bound = overhead + 20
    # sample_paths[s][i] is the i'th sampled path in state s
    sample_paths = [[[init_path] for x in range(num_samples)]]
    # probs = np.ones(sample_paths.shape())
    for y in range(num_states - 1):  # Allstates except the first one
        sample_paths.append([])  # for s=1,...2(k-1), the init path has no districts
    for i in range(2, len(cont_sections) + 1):  # Range is off because negative values are offset
        cur_sect = cont_sections[-i]
        cur_dict = [collections.defaultdict() for x in range(num_states)]
        for s in range(len(cur_dict)):  # state can be 0,1,...,num_states
            # Assumes len(cur_sect)==1, which should always be true, but code is built more generally
            create_labellings_one_section(cur_dict[s], cur_sect[0], s)
        face = face_list[-i + 1]
        label_inds = []  # Inds in flattened_sections
        labeled_edges = []  # List of edges that have labels to make searching later easier
        new_loc = []  # The list of edges that will be added
        # Find index of step
        flattened_sections = [tuple(sorted(x)) for j in range(len(cur_sect)) for x in cur_sect[j]]
        prev_flattened_sections = [tuple(sorted(x)) for j in range(len(prev_sect)) for x in prev_sect[j]]
        inds_to_add = []  # Keep track of which indices of PREV_flattened_sections we need to add to
        exit_locs = []
        index = sum([len(cur_sect[j]) for j in range(len(cur_sect))])
        for j in range(len(face)):
            edge = (face[j], face[((j + 1) % len(face))])
            named_edge = tuple(sorted(edge))
            # edge = (face[((j + 1) % len(face))], face[j])  # We know it'll be reversed!
            if named_edge in outer_boundary:
                # Allow edge to be an exit_edge
                exit_locs.append(named_edge)
            elif named_edge in flattened_sections:
                labeled_edges.append(named_edge)
                cur_index = flattened_sections.index(named_edge)
                label_inds.append(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append(named_edge)
                if named_edge in prev_flattened_sections:  # Need to account for the autohealing in cont_sections
                    inds_to_add.append(prev_flattened_sections.index(named_edge))
        label_inds = sorted(label_inds, reverse=True)
        # if len(set(new_loc[edge_ind-1]).intersection(set(new_loc[edge_ind]))) != 1:
        # Create mapping from paths in cur_dict to those in prev_dict using similar edges in flattened_sections
        trimmed_prev_flattened_sections = [x for x in prev_flattened_sections if x not in new_loc]
        mapping = [trimmed_prev_flattened_sections.index(flattened_sections[j]) for j in range(len(flattened_sections))
                   if flattened_sections[j] not in labeled_edges]
        # need to invert mapping, might be faster to do it above but speed doesnt matter in this part
        mapping = [mapping.index(x) for x in range(len(mapping))]
        # A mapping of paths in prev_dict to their "neighbors" in cur_dict to help with sampling
        path_map = [collections.defaultdict(list) for x in range(len(cur_dict))]
        print("Working on section {0} with length {1}".format(len(cont_sections) - i,
                                                              len(flattened_sections))) if debug else ""
        print("Current face: " + str(face)) if debug else ""
        print("New location: " + str(new_loc)) if debug else ""
        print("Label_inds: " + str(label_inds)) if debug else ""
        for s in range(num_states):
            # prev_dict has the paths from last iteration, which are the consequences of those in cur_dict
            # s is the state of the paths in cur_dict, so path_map always maps prev_s=s(-1?) to others
            # For general k, this may be more than -1: A path can split to at most len(face)-1 directions, which would
            # jump len(face)-2.
            for path in cur_dict[s].keys():
                # Find step type (labels)
                labels = tuple([path[x] for x in label_inds if path[x] != '0'])
                if len(labels) > 2:  # Too many paths meet, just continue
                    continue
                next_path = ''.join([path[x] for x in range(len(path)) if x not in label_inds])
                next_path = ''.join([next_path[mapping[x]] for x in range(len(next_path))])
                # Collect all possible consequences of labels to cur_dict
                if len(labels) == 0:
                    # Can't have any edge exit if there is no 1 label
                    path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                    # path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                    if path1 in prev_dict[s]:
                        cur_dict[s][path] += prev_dict[s][path1]
                        path_map[s][path1].append((path, s))
                    for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                        string_to_add = '0' * ind1 + '3' + '0' * (ind2 - ind1 - 1) + '2' + '0' * (len(new_loc) - ind2 - 1)
                        path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                        # path1 = next_path[:index] + string_to_add + next_path[index:]
                        if path1 in prev_dict[s]:
                            cur_dict[s][path] += prev_dict[s][path1]
                            path_map[s][path1].append((path, s))
                    # If this is surrounded by a 3 and a 2, can we add either a 3-2 or a 2-3?
                    # NO! A 3-2 corresponds to the two paths meeting, while a 2-3 would be a self-intersection, despite
                    # the motzkin path being valid - this is a correct death of a path
                    if s > 0:
                        # Add a potential entrance
                        for exit_ind in range(len(exit_locs)):
                            for ind1 in range(len(new_loc)):
                                string_to_add = '0' * ind1 + '1' + '0' * (len(new_loc) - 1 - ind1)
                                path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                if path1 in prev_dict[s - 1]:
                                    cur_dict[s][path] += prev_dict[s-1][path1]
                                    path_map[s - 1][path1].append((path, s))
                elif len(labels) == 1:
                    for ind1 in range(len(new_loc)):  # Path could just continue in some direction
                        string_to_add = '0' * ind1 + labels[0] + '0' * (len(new_loc) - 1 - ind1)
                        path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                        # path1 = next_path[:index] + string_to_add + next_path[index:]
                        if path1 in prev_dict[s]:
                            cur_dict[s][path] += prev_dict[s][path1]
                            path_map[s][path1].append((path, s))
                    if s > 0:
                        if '1' in labels:
                            # TODO: Create opportunities for higher degree splits (up to len(face)-2)
                            # would have to adjust sampling algorithm and cause some slowdown
                            # Add a potential fork
                            for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                                string_to_add = '0' * ind1 + '1' + '0' * (ind2 - ind1 - 1) + '1' + '0' * (
                                        len(new_loc) - ind2 - 1)
                                path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                # path1 = next_path[:index] + string_to_add + next_path[index:]
                                if path1 in prev_dict[s - 1]:
                                    cur_dict[s][path] += prev_dict[s-1][path1]
                                    path_map[s-1][path1].append((path, s))
                        # Add a potential exit!
                        # index is either 0 or len(path1)-1
                        for exit_ind in range(len(exit_locs)):
                            path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                            count = 0
                            if '1' in labels:  # Simply add 0s into new_loc
                                pass
                            elif '3' in labels:  # Find the next 2 to turn into a 1 (can't be covered bc exit edge)
                                for x in range(index, len(path1)):
                                    if path1[x] == '3':
                                        count += 1
                                    if path1[x] == '2':
                                        if count != 0:
                                            count -= 1
                                        else:
                                            path1 = path1[:x] + '1' + path1[x + 1:]
                                            break
                            elif '2' in labels:  # Find the next 3 to turn into a 1 (can't be covered bc exit edge)
                                for x in range(index - 1, -1, -1):
                                    if path1[x] == '2':
                                        count += 1
                                    if path1[x] == '3':
                                        if count != 0:
                                            count -= 1
                                        else:
                                            path1 = path1[:x] + '1' + path1[x + 1:]
                                            break
                            # Code to find unpaired 3 or 2:
                            # if '1' not in labels:
                            #     for x in range(0, len(path1)):
                            #         true_x = index - 1 - x if x + index >= len(path1) else x + index
                            #         if path1[true_x] == '1':
                            #             path1 = path1[:true_x] + ('3' if '2' in labels else '2') + path1[true_x + 1:]
                            #             break
                            if path1 in prev_dict[s - 1]:
                                cur_dict[s][path] += prev_dict[s - 1][path1]
                                path_map[s - 1][path1].append((path, s))
                elif labels in [('1', '2'), ('2', '1'), ('1', '3'), ('3', '1'), ('2', '2'), ('3', '3'), ('3', '2'), ('1', '1')]:
                    path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                    # path1 = insert_at_indices(path1, '1'*len(sp_inds_to_add), sp_inds_to_add) if len(sp_inds_to_add) > 0 else path1
                    # path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                    count = 0
                    # The order of 3-2's and 2-3's change depending on state-- use ones to represent order change
                    if labels == ('3', '2'):  # possible, just combine
                        pass
                    elif labels == ('1', '1'):  # Two ones meet- can either just merge or branch out
                        if s > 0:  # Don't need upper bound, only looking at smaller previous states
                            # Might be able to merge-- only merge to one for now TODO: make it more
                            for ind1 in range(len(new_loc)):
                                string_to_add = '0' * ind1 + '1' + '0' * (len(new_loc) - 1 - ind1)
                                path2 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                if path2 in prev_dict[s - 1]:
                                    cur_dict[s][path] += prev_dict[s-1][path2]
                                    path_map[s-1][path2].append((path, s))
                        # Either way, we also allow for the 1's to meet without changing the state
                    # Need to find partner and change label:
                    elif '3' in labels:  # 2 will be below it
                        for x in range(index, len(path1)):
                            if path1[x] == '3':
                                count += 1
                            if path1[x] == '2':
                                if count != 0:
                                    count -= 1
                                else:
                                    path1 = path1[:x] + ('1' if labels != ('3', '3') else '3') + path1[x + 1:]
                                    break
                    else:
                        for x in range(index - 1, -1, -1):
                            if path1[x] == '2':
                                count += 1
                            if path1[x] == '3':
                                if count != 0:
                                    count -= 1
                                else:
                                    path1 = path1[:x] + ('1' if labels != ('2', '2') else '2') + path1[x + 1:]
                                    break
                    if count != 0:
                        raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")
                    if path1 in prev_dict[s]:
                        cur_dict[s][path] += prev_dict[s][path1]
                        path_map[s][path1].append((path, s))
                    # if path1 not in prev_dict:
                    #     print(path1)
                # labels is reversed from the path string due to orientation of faces - NOT ALWAYS???
                elif labels == ('2', '3'):  # or labels == ('2', '3'):
                    pass  # Would close a loop, this parent gets no children
                    # print("Closed a loop!")
                    # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
                    # return 0
                    # We just closed a loop! Currently allowed
                    # raise Exception("Theoretically impossible case occurred, we closed a loop.")
                else:
                    raise Exception("Invalid labels on step location")
        # Sample by stepping backwards - this is only possible once we've already created both prev and cur dicts
        # Sol: keep track of PARTIAL full trees, and sample once space is too large
        ##################################################################################
        # This might work, but stepping backwards is nontrivial - for now, slow down above
        # new_sample_paths = [[] for x in range(len(cur_dict))]
        # for cur_k in range(len(cur_dict)):
        #     # offset = 0
        #     for i2 in range(len(sample_paths[cur_k])):
        #         sample_path = sample_paths[cur_k][i2]
        #         count_arr = [cur_dict[x[1]][x[0]] for x in path_map[cur_k][sample_path[-1]]]
        #         if len(count_arr) == 0:
        #             # sample_paths[cur_k].pop(i2 - offset)
        #             # offset += 1
        #             print("Could not continue path: " + str(sample_path))
        #             continue
        #         choice = np.random.uniform(0, sum(count_arr))
        #         sample_ind = np.arange(len(count_arr))[np.asanyarray([sum(count_arr[:x+1]) >= choice for x in range(len(count_arr))])][0]
        #         path_k = path_map[cur_k][sample_path[-1]][sample_ind][1]
        #         new_sample_paths[path_k].append(sample_paths[cur_k][i2])
        #         new_sample_paths[path_k][-1].append(path_map[cur_k][sample_path[-1]][sample_ind][0])
        #         # if path_k != cur_k:
        #         #     sample_paths[path_k].append(sample_paths[cur_k][i2 - offset])
        #         #     sample_paths[path_k][-1].append(path_map[cur_k][sample_path[-1]][sample_ind][0])
        #         #     sample_paths[cur_k].pop(i2-offset)
        #         #     offset += 1
        #         # else:
        #         #     sample_paths[path_k][i2 - offset].append(path_map[cur_k][sample_path[-1]][sample_ind][0])
        # sample_paths = new_sample_paths
        print("i={0}/{1}, {2} paths remain".format(i, len(face_order), sum(len(k_paths) for k_paths in sample_paths))) # if i % 10 == 5 else ""
        # prev_dict = cur_dict
        # prev_sect = cur_sect
        # continue
        #######################################################################
        # Sample tree becomes a sample forest with k+1 trees
        for s in range(len(cur_dict)):
            # Build another layer of sample_tree
            new_layer = collections.defaultdict(list)
            # For each path in the last layer of the sample _forest_
            for prev_s in range(max(0, s-1), s+1):  # k can only increase by 1 TODO: Won't be true
                for p in sample_tree[prev_s][-1 if prev_s == s else -2]:  # sample_tree[prev_s] already has a new layer
                    for m in path_map[prev_s][p]:  # Loop through each tuple of next_path, next_state
                        if m[1] != s:
                            # This next path does not come to this new state
                            continue
                        sample_tree[prev_s][-1 if prev_s == s else -2][p][1].append(m)  # Add the newest paths to the sample tree entry
                        # Add the empty entry into the new layer
                        new_layer[m[0]] = [cur_dict[m[1]][m[0]], []]  # not always cur_s!
            sample_tree[s].append(new_layer)
        # clean empty paths
        # Current issue: prev_rem does not go in between states!
        # Loop through each layer of each sample tree bottom up
        prev_rem = [[] for x in range(num_states)]
        for i2 in range(len(sample_tree[0]) - 1):  # Uses that each state in sample_tree has the same depth
            to_rem = [[] for x in range(num_states)]
            for s in range(len(cur_dict)):
                cur_layer = sample_tree[s][-i2 - 2]  # for i2=0, this is the second to last
                # next_layer = sample_tree[cur_k][-i2 - 1]  # for i2=0, this is the last layer
                # nexter_layer = sample_tree[min(k, cur_k + 1)][-i2 - 1]  # for i2=0, this is the last layer
                # TODO: Make this more than just s+1 for the next layer
                # Note: can optimize this a little when s=num_states-1
                next_layer = set(sample_tree[s][-i2 - 1].keys()).union(set(sample_tree[min(num_states - 1, s + 1)][-i2 - 1].keys()))
                for p in cur_layer:  # p is a tuple of (count, [(next_path, next_s)])
                    offset = 0
                    neighbors = cur_layer[p][1]
                    for j in range(len(neighbors)):
                        child = neighbors[j - offset]
                        if child in prev_rem[s] or (s > 0 and child in prev_rem[s-1]) or child[0] not in next_layer:
                            neighbors.pop(j - offset)
                            offset += 1
                    if len(neighbors) == 0:
                        to_rem[s].append(p)
            for s in range(len(cur_dict)):
                for p in to_rem[s]:
                    sample_tree[s][-i2 - 2].pop(p)
            prev_rem = copy.deepcopy(to_rem)
        tree_size = sum(len(x) for x in sample_tree)
        # Tree is trimmed, just need to sample
        if tree_size >= num_states*subtree_bound or i == len(cont_sections):
            sample_paths, sample_tree = sample_from_tree(cur_dict, sample_paths, sample_tree, overhead)
        prev_dict = cur_dict
        prev_sect = cur_sect
    # for path in sample_paths:
    #     path += ['1']
    return list(prev_dict[num_states-1].values())[0], sample_paths  # There should only be one value at the end


def sample_from_tree(cur_dict, sample_paths, sample_tree, overhead):
    new_sample_paths = [[] for x in range(len(cur_dict))]
    offset = overhead if len(sample_paths[2]) > 1 else 0
    # Sample each of the paths
    # Loop through each initial s > 0
    for cur_s in range(len(cur_dict)):
        # Loop through each path
        for i2 in range(len(sample_paths[cur_s])):
            path_s = cur_s  # The path should start at a certain k
            sample_path = sample_paths[cur_s][i2]
            kill = False
            # Loop through each layer - tree size and sample path length are off by overhead
            for ind in range(len(sample_tree[path_s]) - offset - 1):  # each tree should have the same length
                ind = ind + offset
                sampled_pair = sample_one(sample_path, sample_tree, ind, path_s)  # sample a step (with tracebacks!)
                if sampled_pair is None:
                    # print("This path died at index {0}: {1}".format(ind, sample_path[-4:]))
                    kill = True
                    break
                sample_path, path_s = sampled_pair
            new_sample_paths[path_s].append(sample_path) if not kill else ""  # Add newly sampled path
    sample_paths = new_sample_paths
    # New paths have been sampled in sample_paths
    # reset sample tree to newest path ends plus some overhead
    new_tree = [[(0, dict())] for x in range(len(cur_dict))]
    # Build inital layer from sample_paths
    for s in range(len(cur_dict)):
        new_layer = collections.defaultdict(list)
        # For each path in sample_paths
        for i2 in range(len(sample_paths[s])):
            # Add the current entries
            for state_offset in range(s+1):  # The initial state is at s-state_offset
                # Check if sample_path[-overhead-1] is in the correct layer of state s-offset of sample_tree
                if sample_paths[s][i2][-overhead-1] in sample_tree[s-state_offset][-overhead-1]:
                    # Just copies the elements in sample_paths[..][-overhead-1] over to a new layer, and sets that as new_tree
                    # The -overhead-1 might have been in a different path_k, because sample_paths is indexed by the last path_k
                    # Potentially even s steps, up to overhead. Indexing is incredibly awkward
                    new_layer[sample_paths[s][i2][-overhead-1]] = \
                        sample_tree[s-state_offset][-overhead-1][sample_paths[s][i2][-overhead-1]]
            # else:
            #     print("Path {0} failed inexplicably. cur_k is {1}".format(sample_paths[s][i2], s))
        new_tree[s] = [new_layer]
    # Rebuild tree from whatever is left below the overhead
    for level in range(overhead):
        for s in range(len(cur_dict)):
            # Build another layer of sample_tree
            new_layer = collections.defaultdict(list)
            # For each path in the last layer of the sample tree
            for prev_s in range(max(0, s - 1), s + 1):  # s can only increase by 1 TODO: make bigger
                for v in new_tree[prev_s][-1 if prev_s == s else -2].values():  # sample_tree[prev_k] already has a new layer
                    # There is a strange case of empty entries coming from the fact that sample_tree is a default_dict,
                    # and if we ever query for a nonexisting entry it creates and empty list. This should not be
                    # happening, need to find mistake and fix it. For now this is a workaround.
                    if len(v) == 0:  # happens first at i=64???
                        continue
                    # p is of the form [count, {(next_paths, next_states)}]
                    # Add the entry from sample_tree to new_tree
                    for m in v[1]:  # m is of the form (next_path, next_state)
                        if m[1] == s:  # m goes to this current state
                            if m[0] in sample_tree[s][-overhead+level]:
                                # Add the (potentially) empty entry into the new layer
                                new_layer[m[0]] = sample_tree[s][-overhead+level][m[0]]
                            else:
                                print("This should not happen. Avoiding creating an empty entry, but debug this.")
            new_tree[s].append(new_layer)
    return sample_paths, new_tree


# Returns a sampled path of length cur_ind modulo tree size cutoff
def sample_one(sample_path, sample_tree, cur_ind, path_k):
    layer = sample_tree[path_k][cur_ind]
    if sample_path[-1] not in layer:  # Try to take a step back and salvage it?
        if cur_ind > 0:
            # path_k may have changed as we step back-- loop through all and ensure the step is in the right direction
            for prev_k in [path_k, path_k-1]:  # start with path_k
                if prev_k < 0:
                    continue
                new_path = sample_one(sample_path[:-1], sample_tree, cur_ind - 1, prev_k)
                if new_path is not None:
                    sample_path, path_k = new_path
                    break  # Might mess with uniformity a little?
            if new_path is None:
                return None  # recursively kill off path
        else:
            return None  # End case
    layer = sample_tree[path_k][cur_ind]  # May have modified path_k, need to redefine layer
    if sample_path[-1] not in layer:  # layer may have change, so we need to recheck if next step is possible
        return None  # Just give up as we always prefer keeping the same path_k, no way to salvage it
    # count of sample_tree[next_k][next_layer][next_path] for (next_path, next_k) in layer[path[-1]]
    # count_arr = [sample_tree[x[1]][ind + 1][x[0]][0] for x in layer[sample_path[-1]][1]]
    # Isn't it always uniform anyways? Not quite- counts for children can come from multiple parents
    # Or just make it uniform
    count_arr = [1 for x in layer[sample_path[-1]][1] if len(x) > 0]
    if len(count_arr) == 0:
        return None  # Really should never happen, unless layer has an empty entry
    choice = np.random.uniform(0, sum(count_arr))
    sample_ind = np.arange(len(count_arr))[
        np.asanyarray([sum(count_arr[:x + 1]) >= choice for x in range(len(count_arr))])][0]
    # Set new path_k and append next step in path
    path_k = layer[sample_path[-1]][1][sample_ind][1]
    # if path_k != prev_path_k:
    #     sample_paths[path_k].append(sample_paths[prev_path_k][i2-offset])
    #     sample_paths[path_k][-1].append(layer[sample_path[-1]][1][sample_ind][0])
    #     offset += 1
    # else:
    #     sample_paths[path_k][i2 - offset].append(layer[sample_path[-1]][1][sample_ind][0])
    sample_path.append(layer[sample_path[-1]][1][sample_ind][0])
    return sample_path, path_k


def create_labellings_multisection(cur_dict, cur_sect, flat_outer):
    for tup in partition(depth_bound, len(cur_sect)):  # Loop through all ordered partitions for the depth bound
        temp_dict = collections.defaultdict()  # Need a separate dictionary for each loop...
        temp_dict[''] = 0
        # if len(tup) > len(cur_sect):
        #     continue
        # if len(tup) < len(cur_sect):
        #     tup += (0, ) * (len(cur_sect) - len(tup))
        for section_ind in range(len(cur_sect)):
            section = cur_sect[section_ind]
            # Generate all possible motzkin paths for cur_sect
            # find section that connects to start_edge
            is_first = False  # be more careful if you actually have to do this
            # if not (len(cur_sect) == 1 and i > len(outer_boundary)):
            if is_first:
                my_dict = {}
                find_motzkin_paths(0, '', len(section), my_dict, depth_bound - tup[section_ind], with_one=True)
                temp_dict = {s1 + s2: 0 for s1 in temp_dict.keys() for s2 in my_dict.keys()}
            else:
                my_dict = {}
                find_motzkin_paths(0, '', len(section), my_dict, depth_bound - tup[section_ind], with_one=False)
                temp_dict = {s1 + s2: 0 for s1 in temp_dict.keys() for s2 in my_dict.keys()}
        cur_dict.update(temp_dict)
    # Add labellings that swap 2's above and 3's below with 1's:
    my_dict = {}
    for s in cur_dict:
        if '1' not in s:
            continue
        one_ind = s.index('1')
        for j in range(len(s)):
            if s[j] != '0':
                if j < one_ind and s[j] == '2':
                    new_s = s[:j] + '1' + s[j + 1:one_ind] + s[j] + s[one_ind + 1:]
                    my_dict[new_s] = 0
                if j > one_ind and s[j] == '3':
                    new_s = s[:one_ind] + s[j] + s[one_ind + 1:j] + '1' + s[j + 1:]
                    my_dict[new_s] = 0
    cur_dict.update(my_dict)


# Generates all possible Motzkin labellings for a boundary with cur_state=k
def create_labellings_one_section(cur_dict, cur_sect, k):
    temp_dict = collections.defaultdict()  # Need a separate dictionary for each loop...
    temp_dict[''] = 0
    # for section_ind in range(len(cur_sect)):
    # Generate all possible motzkin paths for cur_sect
    my_dict = {}
    for x in range(k % 2, k+2, 2):  # All smaller numbers with the same parity
        find_motzkin_paths(0, '', len(cur_sect), my_dict, 0, x)
    cur_dict.update(my_dict)
    # Add labellings that swap 2's above and 3's below with 1's:
    my_dict = {}
    for s in cur_dict:
        if '1' not in s:
            continue
        one_ind = s.index('1')
        for j in range(len(s)):
            if s[j] != '0':
                if j < one_ind and s[j] == '2':
                    new_s = s[:j] + '1' + s[j + 1:one_ind] + s[j] + s[one_ind + 1:]
                    my_dict[new_s] = 0
                if j > one_ind and s[j] == '3':
                    new_s = s[:one_ind] + s[j] + s[one_ind + 1:j] + '1' + s[j + 1:]
                    my_dict[new_s] = 0
    cur_dict.update(my_dict)


# https://doi.org/10.1016/j.tcs.2020.12.013
# Returns _every_ possible Motzkin path with length n, prefix w, height h, and max_depth depth.
# @param h: The current height of the path
# @param w: The current prefix word
# @param n: The total length of the path
# @param m_dict: The dictionary holding the output paths
# @param depth: The max height the path can reach
# @param num_ones: The number of ones the path must have
def find_motzkin_paths(h, w, n, m_dict, depth, num_ones):
    j = len(w)
    if depth > depth_bound:  # Too deep
        return
    if h > n - j:
        return
    if j > n and num_ones == 0:  # Done
        m_dict[w] = 0
        return
    if h == n - j and num_ones == 0:  # Fill the rest with 2's
        m_dict[w + h * '2'] = 0
        return
    if h > 0:
        find_motzkin_paths(h - 1, w + '2', n, m_dict, depth, num_ones)  # can add a two!
    if h == 0 and num_ones > 0:  # Can add a one too!
        find_motzkin_paths(h, w + '1', n, m_dict, depth, num_ones - 1)  # can have one less one
    find_motzkin_paths(h, w + '0', n, m_dict, depth, num_ones)
    find_motzkin_paths(h + 1, w + '3', n, m_dict, depth + 1, num_ones)
    return


def count_non_int_paths_unrestr(face_list, start_edge, outer_boundary, cont_sections):
    # Loop through reversed cont_sections - keep track of cur_sect and prev_sect
    # Generate and loop through each motzkin path of cur_sect and find connected path in prev_sect
    # Add the values of each of cur_sect path's neighbors from prev_sect to it's value
    # Store these in prev_dict and cur_dict
    # prev_dict = cur_dict - manually gc.collect() after?
    prev_dict = {'1': 1}  # Do we need to store which step it is in these dicts? No!
    prev_sect = cont_sections[-1]
    flat_outer = [x[0] for x in outer_boundary] + [outer_boundary[-1][1]]
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    start_edge_ind = outer_boundary.index(start_edge)
    for i in range(2, len(cont_sections) + 1):  # Range is off because negative values are offset
        cur_sect = cont_sections[-i]
        cur_dict = collections.defaultdict()
        cur_dict[''] = 0
        # gc.collect()
        for section in cur_sect:
            # Generate all possible motzkin paths for cur_sect
            # find section that connects to start_edge
            is_first = False
            if not (len(cur_sect) == 1 and i > len(outer_boundary)):
                start_ind = flat_outer.index(section[0][0]) if section[0][0] in flat_outer else flat_outer.index(
                    section[0][1])
                end_ind = flat_outer.index(section[-1][1]) if section[-1][1] in flat_outer else flat_outer.index(
                    section[-1][0])
                is_first = start_ind <= start_edge_ind <= end_ind
            is_first = is_first or (len(cur_sect) == 1 and i > len(outer_boundary))
            if is_first:
                my_dict = {}
                for one_ind in range(len(section)):
                    first_dict = {}
                    second_dict = {}
                    find_motzkin_paths_unrestr(0, '', len(section) - one_ind - 1, first_dict)
                    find_motzkin_paths_unrestr(0, '', one_ind, second_dict)
                    my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                cur_dict = {s1 + s2: 0 for s1 in cur_dict.keys() for s2 in my_dict.keys()}
            else:
                my_dict = {}
                find_motzkin_paths_unrestr(0, '', len(section), my_dict)
                cur_dict = {s1 + s2: 0 for s1 in cur_dict.keys() for s2 in my_dict.keys()}
        face = face_list[-i + 1]
        label_inds = []  # Inds in flattened_sections
        labeled_edges = []  # List of edges that have labels to make seraching later easier
        new_loc = []  # The list of edges that will be added
        # Find index of step
        flattened_sections = [tuple(sorted(x)) for j in range(len(cur_sect)) for x in cur_sect[j]]
        prev_flattened_sections = [tuple(sorted(x)) for j in range(len(prev_sect)) for x in prev_sect[j]]
        inds_to_add = []  # Keep track of which indices of PREV_flattened_sections we need to add to
        index = sum([len(cur_sect[j]) for j in range(len(cur_sect))])
        for j in range(len(face)):
            edge = (face[j], face[((j + 1) % len(face))])
            named_edge = tuple(sorted(edge))
            # edge = (face[((j + 1) % len(face))], face[j])  # We know it'll be reversed!
            if named_edge in outer_boundary:
                pass
            elif named_edge in flattened_sections:
                labeled_edges.append(named_edge)
                cur_index = flattened_sections.index(named_edge)
                label_inds.append(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append(named_edge)
                if named_edge in prev_flattened_sections:  # Need to account for the autohealing in cont_sections
                    inds_to_add.append(prev_flattened_sections.index(named_edge))
        # Create mapping from paths in cur_dict to those in prev_dict using similar edges in flattened_sections
        trimmed_prev_flattened_sections = [x for x in prev_flattened_sections if x not in new_loc]
        mapping = [trimmed_prev_flattened_sections.index(flattened_sections[j]) for j in range(len(flattened_sections))
                   if flattened_sections[j] not in labeled_edges]
        mapping = [mapping.index(x) for x in range(
            len(mapping))]  # need to invert mapping, might be faster to do it above but speed doesnt matter in this part
        print("Working on section {0} with length {1}".format(len(cont_sections) - i,
                                                              '.'.join([str(len(sect)) for sect in cur_sect])))
        print("Current face: " + str(face)) if debug else ""
        print("New location: " + str(new_loc)) if debug else ""
        print("Label_inds: " + str(label_inds)) if debug else ""
        for path in cur_dict.keys():
            # Find step type (labels)
            labels = tuple([path[x] for x in label_inds if path[x] != '0'])
            if len(labels) > 2:  # Too many paths meet, just continue
                continue
            next_path = ''.join([path[x] for x in range(len(path)) if x not in label_inds])
            next_path = ''.join([next_path[mapping[x]] for x in range(len(next_path))])
            # Collect all possible consequences of labels to cur_dict
            if len(labels) == 0:
                path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                # path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
                for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                    string_to_add = '0' * ind1 + '3' + '0' * (ind2 - ind1 - 1) + '2' + '0' * (len(new_loc) - ind2 - 1)
                    path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                    # path1 = next_path[:index] + string_to_add + next_path[index:]
                    cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
            elif len(labels) == 1:
                for ind1 in range(len(new_loc)):
                    string_to_add = '0' * ind1 + labels[0] + '0' * (len(new_loc) - 1 - ind1)
                    path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                    # path1 = next_path[:index] + string_to_add + next_path[index:]
                    cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
            elif labels in [('1', '2'), ('2', '1'), ('1', '3'), ('3', '1'), ('2', '2'), ('3', '3'), ('2', '3')]:
                path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                # path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                count = 0
                if labels == ('2', '3'):  # possible, just combine
                    pass
                # Need to find partner and change label:
                elif '3' in labels:  # 2 will be below it
                    for x in range(index, len(path1)):
                        if path1[x] == '3':
                            count += 1
                        if path1[x] == '2':
                            if count != 0:
                                count -= 1
                            else:
                                path1 = path1[:x] + ('1' if labels != ('3', '3') else '3') + path1[x + 1:]
                                break
                else:
                    for x in range(index, -1, -1):
                        if path1[x] == '2':
                            count += 1
                        if path1[x] == '3':
                            if count != 0:
                                count -= 1
                            else:
                                path1 = path1[:x] + ('1' if labels != ('2', '2') else '2') + path1[x + 1:]
                                break
                if count != 0:
                    raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")
                cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
            elif labels == ('3', '2'):
                pass  # ?
                # print("Closed a loop!")
                # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
                # return 0
                # We just closed a loop! Currently allowed
                # raise Exception("Theoretically impossible case occurred, we closed a loop.")
            else:
                raise Exception("Invalid labels on step location")
        prev_dict = cur_dict
        prev_sect = cur_sect
    return list(prev_dict.values())[0]  # There should only be one value at the end


# https://doi.org/10.1016/j.tcs.2020.12.013
def find_motzkin_paths_unrestr(h, w, n, m_dict):
    j = len(w)
    if h > n - j:
        return
    if j > n:
        m_dict[w] = 0
        return
    if h == n - j:
        m_dict[w + h * '2'] = 0
        return
    if h > 0:
        find_motzkin_paths_unrestr(h - 1, w + '2', n, m_dict)
    find_motzkin_paths_unrestr(h, w + '0', n, m_dict)
    find_motzkin_paths_unrestr(h + 1, w + '3', n, m_dict)
    return


def allocate_table(face_list, outer_boundary, cont_sections, k, conn: sqlite3.Connection=None, just_sample=False):
    if just_sample:  # conn must then not be None
        cur = conn.cursor()
        cur.execute("select * from sqlite_master where type='table' and tbl_name like 'nodes_%';")
        rows = cur.fetchall()
        return [[],] * len(rows), None
    # Put the one in first
    # How should I keep track of contiguous sections? - build up
    # How to find edge relations? - top down and reverse
    # How should I enumerate all Motzkin paths? - Use recursive method to create strings for each section and then
    # take the cartesian product
    big_table = []
    num_states = 2 * (k - 1) + 1  # The number of possible states the boundary can be at
    if conn is not None:
        cur = conn.cursor()
        # cur.execute("CREATE TABLE IF NOT EXISTS nodes_0 (node_name text PRIMARY KEY, _count integer);")
        # for s in range(num_states):
        #     # This will store the final count
        #     cur.execute("INSERT INTO nodes_0 (node_name, _count) VALUES (?, 0)", ('.' + str(s),))
        # conn.commit()
        # big_table.append([])  # for counting
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    prev_dict = None
    for i in range(len(cont_sections)):
        cur_sect = cont_sections[i]
        path_dict = [collections.defaultdict() for x in range(num_states)]
        if conn is not None:
            sql_create_table = """CREATE TABLE IF NOT EXISTS {0} (
                                                                node_name text PRIMARY KEY,
                                                                _count integer
                                                            ); """.format("nodes_" + str(i))
            cur.execute(sql_create_table)
            conn.commit()
            big_table.append([])  # for counting
        else:
            big_table.append(path_dict)
        # gc.collect()
        for s in range(len(path_dict)):  # state can be 0,1,...,num_states
            # Assumes len(cur_sect)==1, which should always be true, but code is built more generally
            create_labellings_one_section(path_dict[s], cur_sect[0], s)
            if conn is not None:
                sql_insert = "INSERT INTO {0}(node_name,_count) VALUES(?,?)".format("nodes_" + str(i))
                for name in path_dict[s].keys():
                    cur.execute(sql_insert, [name + "." + str(s), 0])
                conn.commit()
        if i % 10 == 0:
            print("Populating table entry {0}/{1}".format(i, len(cont_sections)))
    end_layer = [collections.defaultdict() for x in range(num_states)]
    end_layer[-1][''] = 1
    big_table.append(end_layer) if conn is None else ""
    print("Succesfully generated motzkin paths.")
    # Generate edge relations
    edge_maps = []  # maintain a list of each layer's path_maps
    for i in range(1, len(cont_sections)):
        if i % 10 == 0:
            print("Generating path maps for layer {0}/{1}".format(i, len(cont_sections)))
        next_sect = cont_sections[i]
        prev_sect = cont_sections[i-1]# if i > 0 else [[]]
        face = face_list[i-1]
        label_inds = []  # Inds in flattened_sections
        labeled_edges = []  # List of edges that have labels to make searching later easier
        new_loc = []  # The list of edges that will be added
        # Find index of step
        prev_flattened_sections = [tuple(sorted(x)) for j in range(len(prev_sect)) for x in prev_sect[j]]
        next_flattened_sections = [tuple(sorted(x)) for j in range(len(next_sect)) for x in next_sect[j]]
        inds_to_add = []  # Keep track of which indices of NEXT_flattened_sections we need to add to
        exit_locs = []
        index = sum([len(next_sect[j]) for j in range(len(next_sect))])
        for j in range(len(face)):
            edge = (face[j], face[((j + 1) % len(face))])
            named_edge = tuple(sorted(edge))
            # edge = (face[((j + 1) % len(face))], face[j])  # We know it'll be reversed!
            if named_edge in outer_boundary:
                # Allow edge to be an exit_edge
                exit_locs.append(named_edge)
            elif named_edge in prev_flattened_sections:
                labeled_edges.append(named_edge)
                cur_index = prev_flattened_sections.index(named_edge)
                label_inds.append(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append(named_edge)
                if named_edge in next_flattened_sections:  # Need to account for the autohealing in cont_sections
                    inds_to_add.append(next_flattened_sections.index(named_edge))
        label_inds = sorted(label_inds, reverse=True)
        # Create mapping from paths in prev_dict to those in next_dict using similar edges in flattened_sections
        trimmed_next_flattened_sections = [x for x in next_flattened_sections if x not in new_loc]
        mapping = [trimmed_next_flattened_sections.index(prev_flattened_sections[j]) for j in range(len(prev_flattened_sections))
                   if prev_flattened_sections[j] not in labeled_edges]
        # need to invert mapping, might be faster to do it above but speed doesnt matter in this part
        mapping = [mapping.index(x) for x in range(len(mapping))]
        # A mapping of paths in next_dict to their "neighbors" in prev_dict that we save in edge_maps
        path_map = [collections.defaultdict(list) for x in range(num_states)]
        print("Working on section {0} with length {1}".format(i, len(next_flattened_sections))) if debug else ""
        print("Current face: " + str(face)) if debug else ""
        print("New location: " + str(new_loc)) if debug else ""
        print("Label_inds: " + str(label_inds)) if debug else ""
        if conn is not None:
            next_dict = [dict() for x in range(num_states)]
            for val, c in cur.execute("SELECT * FROM {0}".format("nodes_" + str(i))):
                p, s = val.split(".")
                next_dict[int(s)][p] = c
            # prev_dict = cur.execute("SELECT * FROM {0}".format("nodes_" + str(i-1))) if i > 0 else list(['.' + str(x) for x in range(num_states)])
            if prev_dict is None:
                prev_dict = [{'': 0} for x in range(num_states)]  # coincidentally still works
        else:
            next_dict = big_table[i]
            prev_dict = big_table[i - 1] if i > 0 else [{'': 0} for x in range(num_states)]
        for s in range(num_states):
            for path in prev_dict[s].keys():
                # Find step type (labels)
                labels = tuple([path[x] for x in label_inds if path[x] != '0'])
                if len(labels) > 2:  # Too many paths meet, just continue
                    continue
                next_path = ''.join([path[x] for x in range(len(path)) if x not in label_inds])
                next_path = ''.join([next_path[mapping[x]] for x in range(len(next_path))])
                # Add edge to all possible consequences of labels
                if len(labels) == 0:
                    # Can't have any edge exit if there is no 1 label
                    path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                    # path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                    if path1 in next_dict[s]:
                        path_map[s][path1].append((path, s))
                    for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                        string_to_add = '0' * ind1 + '3' + '0' * (ind2 - ind1 - 1) + '2' + '0' * (
                                len(new_loc) - ind2 - 1)
                        path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                        # path1 = next_path[:index] + string_to_add + next_path[index:]
                        if path1 in next_dict[s]:
                            path_map[s][path1].append((path, s))
                    # If this is surrounded by a 3 and a 2, can we add either a 3-2 or a 2-3?
                    # NO! A 3-2 corresponds to the two paths meeting, while a 2-3 would be a self-intersection, despite
                    # the motzkin path being valid - this is a correct death of a path
                    if s < num_states-1:
                        # Add a potential entrance
                        for exit_ind in range(len(exit_locs)):
                            for ind1 in range(len(new_loc)):
                                string_to_add = '0' * ind1 + '1' + '0' * (len(new_loc) - 1 - ind1)
                                path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                if path1 in next_dict[s + 1]:
                                    path_map[s + 1][path1].append((path, s))
                elif len(labels) == 1:
                    for ind1 in range(len(new_loc)):  # Path could just continue in some direction
                        string_to_add = '0' * ind1 + labels[0] + '0' * (len(new_loc) - 1 - ind1)
                        path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                        # path1 = next_path[:index] + string_to_add + next_path[index:]
                        if path1 in next_dict[s]:
                            path_map[s][path1].append((path, s))
                    if s < num_states-1:
                        if '1' in labels:
                            # TODO: Create opportunities for higher degree splits (up to len(face)-2)
                            # would have to adjust sampling algorithm and cause some slowdown
                            # Add a potential fork
                            for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                                string_to_add = '0' * ind1 + '1' + '0' * (ind2 - ind1 - 1) + '1' + '0' * (
                                        len(new_loc) - ind2 - 1)
                                path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                # path1 = next_path[:index] + string_to_add + next_path[index:]
                                if path1 in next_dict[s + 1]:
                                    path_map[s + 1][path1].append((path, s))
                        # Add a potential exit!
                        # index is either 0 or len(path1)-1
                        for exit_ind in range(len(exit_locs)):
                            path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                            count = 0
                            if '1' in labels:  # Simply add 0s into new_loc
                                pass
                            elif '3' in labels:  # Find the next 2 to turn into a 1 (can't be covered bc exit edge)
                                for x in range(index, len(path1)):
                                    if path1[x] == '3':
                                        count += 1
                                    if path1[x] == '2':
                                        if count != 0:
                                            count -= 1
                                        else:
                                            path1 = path1[:x] + '1' + path1[x + 1:]
                                            break
                            elif '2' in labels:  # Find the next 3 to turn into a 1 (can't be covered bc exit edge)
                                for x in range(index - 1, -1, -1):
                                    if path1[x] == '2':
                                        count += 1
                                    if path1[x] == '3':
                                        if count != 0:
                                            count -= 1
                                        else:
                                            path1 = path1[:x] + '1' + path1[x + 1:]
                                            break
                            if path1 in next_dict[s + 1]:
                                path_map[s + 1][path1].append((path, s))
                elif labels in [('1', '2'), ('2', '1'), ('1', '3'), ('3', '1'), ('2', '2'), ('3', '3'), ('3', '2'),
                                ('1', '1')]:
                    path1 = insert_at_indices(next_path, '0' * len(new_loc), inds_to_add)
                    count = 0
                    # The order of 3-2's and 2-3's change depending on state-- use ones to represent order change
                    if labels == ('3', '2'):  # possible, just combine
                        pass
                    elif labels == ('1', '1'):  # Two ones meet- can either just merge or branch out
                        if s < num_states-1:
                            # Might be able to merge-- only merge to one for now TODO: make it more
                            for ind1 in range(len(new_loc)):
                                string_to_add = '0' * ind1 + '1' + '0' * (len(new_loc) - 1 - ind1)
                                path2 = insert_at_indices(next_path, string_to_add, inds_to_add)
                                if path2 in next_dict[s + 1]:
                                    path_map[s + 1][path2].append((path, s))
                        # Either way, we also allow for the 1's to meet without changing the state
                    # Need to find partner and change label:
                    elif '3' in labels:  # 2 will be below it
                        for x in range(index, len(path1)):
                            if path1[x] == '3':
                                count += 1
                            if path1[x] == '2':
                                if count != 0:
                                    count -= 1
                                else:
                                    path1 = path1[:x] + ('1' if labels != ('3', '3') else '3') + path1[x + 1:]
                                    break
                    else:
                        for x in range(index - 1, -1, -1):
                            if path1[x] == '2':
                                count += 1
                            if path1[x] == '3':
                                if count != 0:
                                    count -= 1
                                else:
                                    path1 = path1[:x] + ('1' if labels != ('2', '2') else '2') + path1[x + 1:]
                                    break
                    if count != 0:
                        raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")
                    if path1 in next_dict[s]:
                        path_map[s][path1].append((path, s))

                    # if path1 not in prev_dict:
                    #     print(path1)
                # labels is reversed from the path string due to orientation of faces - NOT ALWAYS???
                elif labels == ('2', '3'):  # or labels == ('2', '3'):
                    pass  # Would close a loop, this parent gets no children
                    # print("Closed a loop!")
                    # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
                    # return 0
                    # We just closed a loop! Currently allowed
                    # raise Exception("Theoretically impossible case occurred, we closed a loop.")
                else:
                    raise Exception("Invalid labels on step location")
        if conn is not None:
            sql_create_table = """CREATE TABLE IF NOT EXISTS {0} (
                                                    start_node text,
                                                    end_node text,
                                                    FOREIGN KEY (start_node) REFERENCES {1} (node_name),
                                                    FOREIGN KEY (end_node) REFERENCES {2} (node_name)
                                                ); """.format("map_"+str(i-1), "nodes_"+str(i-1), "nodes_"+str(i))
            cur.execute(sql_create_table)
            conn.commit()
            sql_insert = "INSERT INTO {0}(start_node,end_node) VALUES(?,?)".format("map_" + str(i-1))
            for s in range(len(path_map)):
                for source in path_map[s]:
                    for target in path_map[s][source]:
                        cur.execute(sql_insert, [source + "." + str(s), target[0] + "." + str(target[1])])
            conn.commit()
            prev_dict = next_dict
        else:
            edge_maps.append(path_map)
    # first_layer = [{'': 0} for x in range(num_states)]
    # big_table = [first_layer] + big_table
    return big_table, edge_maps


# Make sure that a face is oriented counterclockwise and starts with the lowest x value
def ensure_ccw(face, positions):
    face_geo = np.asanyarray([positions[x] for x in face])
    min_ind = np.argmin(face_geo, axis=0)[1]  # find the minimum y index
    sgn = np.sign(np.cross(face_geo[(min_ind - 1) % len(face)] - face_geo[min_ind], face_geo[(min_ind + 1) % len(face)]
                           - face_geo[min_ind]))
    min_x_ind = np.argmin(face_geo, axis=0)[0]  # find the minimum x index
    if sgn == 1:
        return [face[(x + min_x_ind) % len(face)] for x in range(len(face))]
    else:
        return [face[(min_x_ind - x) % len(face)] for x in range(len(face))]


# Make sure that a face is oriented counterclockwise and starts with the lowest x value
def ensure_cw(face, positions):
    face_geo = np.asanyarray([positions[x] for x in face])
    min_ind = np.argmin(face_geo, axis=0)[1]  # find the minimum y index
    sgn = np.sign(np.cross(face_geo[(min_ind - 1) % len(face)] - face_geo[min_ind], face_geo[(min_ind + 1) % len(face)]
                           - face_geo[min_ind]))
    min_x_ind = np.argmin(face_geo, axis=0)[0]  # find the minimum x index
    if sgn == -1:
        return [face[(x + min_x_ind) % len(face)] for x in range(len(face))]
    else:
        return [face[(min_x_ind - x) % len(face)] for x in range(len(face))]


# Order and orient each face within face_order in place by traversing each face in g
def orient_faces(face_order, g, positions, start_edge):
    for i in range(len(face_order)):
        f = face_order[i]
        done = False
        for x in range(len(f)):
            for y in range(x+1, len(f)):
                if f[y] in g[f[x]]: # These two vertices of f are connected
                    if generate_face_order.same_face(g.traverse_face(f[x], f[y]), f):
                        face_order[i] = g.traverse_face(f[x], f[y])
                        done = True
                        break
                    elif generate_face_order.same_face(g.traverse_face(f[y], f[x]), f):
                        face_order[i] = g.traverse_face(f[y], f[x])
                        done = True
                        break
            if done:
                break
    # Orient each face in face_order:
    for i in range(len(face_order)):
        face = ensure_cw(face_order[i], positions)
        face_order = face_order[:i] + [face] + face_order[i + 1:]
    # Make sure initial face adheres to the starting edge
    if face_order[0][0] != start_edge[1]:
        face_order[0] = [face_order[0][(x + face_order[0].index(start_edge[1])) % len(face_order[0])] for x in range(len(face_order[0]))]
    return face_order


# Helper function to insert str2 into str1 at locations given by indices
def insert_at_indices(str1, str2, indices):
    offset = 0
    out_str = ''
    for k in range(len(str1) + len(str2)):
        if k in indices:
            out_str += str2[offset]
            offset += 1
        else:
            out_str += str1[k - offset]
    return out_str


# Method to enumerate all possible non-self-intersecting paths crossing a given shapefile and adjacency information
# corresponding to the shapefile.
# Args:
# adj_file - the path to a dbf file generated from arcgis's polygon neighborhood tool that contains the adjacency
# information. NOTE: The graph corresponding to this data should be planar, but may have dangling vertices.
# shapefile - the path to a shp file that contains the local geometry and whatever voter data. For now, this is only
# used to compute centroids which ensure proper orientation for the planar graph.
def enumerate_paths(adj_file, shapefile, recalculate=False, draw=True):
    print("Start time: " + str(time.time()))
    df = gpd.read_file(adj_file)
    np_df = df.to_numpy()
    g_data = collections.defaultdict(list)
    for i in range(len(np_df)):
        g_data[np_df[i][0]].append(np_df[i][1]) if np_df[i][2] > 0.00001 else ""
    loc_df = gpd.read_file(shapefile, driver='ESRI shapefile', encoding='UTF-8')
    loc_df['centroid_column'] = loc_df.centroid
    # centers = loc_df.set_geometry('centroid_column')
    # centers.set_index('OBJECTID', inplace=True)
    # print(centers)
    h = nx.DiGraph(incoming_graph_data=g_data)
    exit_edge = (71, 74)
    start_edge = (46, 48)
    # y_locs = {x: centers.loc[x]['centroid_column'].y for x in h.nodes}
    # stddev = np.std(np.asanyarray(list(y_locs.values())))
    # center = np.mean(
    #     np.asanyarray([y_locs[exit_edge[0]], y_locs[exit_edge[1]], y_locs[start_edge[0]], y_locs[start_edge[0]]]))
    # new_verts = [x for x in h.nodes if math.fabs(y_locs[x] - center) < stddev / 2.25]
    # h2 = h.subgraph(new_verts).copy()
    # g_data = {x: [y for y in g_data[x] if y in new_verts] for x in new_verts}
    while True:
        to_remove = []
        for v, neighbs in g_data.items():
            if len(neighbs) == 1:
                to_remove.append(v)
        if len(to_remove) == 0:
            break
        print(to_remove)
        for v in to_remove:
            g_data[g_data[v][0]].remove(v) if len(g_data[v]) > 0 else ""
            g_data.pop(v)
    positions = nx.planar_layout(h)
    g = nx.PlanarEmbedding()

    # A helper method that returns the angle between a vector x and a basis vect2
    def compare_verts(x, vect2):
        # vect1 = np.array([centers.loc[x]['centroid_column'].x,
        #                   centers.loc[x]['centroid_column'].y]).flatten()
        vect1 = np.array(positions[x])
        vect = vect1 - vect2
        normalized_v = vect / np.linalg.norm(vect)
        return np.ma.arctan2(normalized_v[1], normalized_v[0]) + 2 * np.pi

    # Sort angles w/o computing atan2, slightly faster for a computationally insignificant portion:
    # Input:  dx, dy: coordinates of a (difference) vector.
    # Output: a number from the range [-2 .. 2] which is monotonic
    #         in the angle this vector makes against the x axis.
    #         and with the same discontinuity as atan2
    def pseudoangle(x, vect2):
        vect1 = np.array(positions[x])
        vect = vect1 - vect2
        dx, dy = vect
        p = dx / (abs(dx) + abs(dy))  # -1 .. 1 increasing with x
        if dy < 0:
            return p - 1  # -2 .. 0 increasing with x
        else:
            return 1 - p  # 0 .. 2 decreasing with x

    oriented_g_data = {}
    for v, neighbs in g_data.items():
        # Sort neighbors by orientation of vectors
        # vect2 = np.array([centers.loc[v]['centroid_column'].x,
        #                   centers.loc[v]['centroid_column'].y]).flatten()
        vect2 = np.array(positions[v])
        new_neighb = sorted(neighbs, key=lambda x: pseudoangle(x, vect2), reverse=True)
        # print("{0}: {1}".format(v, new_neighb))
        oriented_g_data[v] = new_neighb
    print(oriented_g_data)
    g.set_data(oriented_g_data)

    # exit(0)
    # positions = nx.combinatorial_embedding_to_pos()
    success, counterexample = nx.check_planarity(g, counterexample=True)
    if not success:
        nx.draw(counterexample, pos=positions, with_labels=True)
        plt.show()
        print("Error: Adjacency graph is not planar, exiting...")
        exit(0)
    if draw:
        # ax = plt.subplot(121)
        # plt.sca(ax)
        plt.figure(figsize=(65, 65))
        nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=8, font_color='red', linewidths=0,
                width=.2)
        # G2 = h.subgraph([106,0,3,4,5,6,7,8,9,14,107,13,51,52])
        # nx.draw(G2, pos=positions, with_labels=True)
        # centers.plot()
        # loc_df.plot()
        plt.show()
        exit()
    g.check_structure()
    print(order_faces(g, positions, start_edge, exit_edge))


def enumerate_paths_with_order(shapefile, face_order, draw=True, recalculate=False):
    print("Start time: " + str(time.time()))
    root = shapefile[:shapefile.index(".")]
    if os.path.exists(root + ".adjlist") and not recalculate:
        print("Using cached adjacency {0}".format(root + ".adjlist"))
        h = nx.read_adjlist(root + ".adjlist", nodetype=int)
        success, g = nx.check_planarity(h, counterexample=True)
        positions = nx.planar_layout(g)
    else:
        # df = gpd.read_file("data/exp2627neighb.dbf")
        # np_df = df.to_numpy()
        # g_data = collections.defaultdict(list)
        # for i in range(len(np_df)):
        #     g_data[np_df[i][0]].append(np_df[i][1]) if np_df[i][2] > 0.00001 else ""
        # Explode the geometries
        gdf = gpd.read_file(shapefile, encoding='UTF-8')
        # cbg_map = pd.read_csv("data/wi_cong_dist/Governor's LC Congressional.csv", sep=',').to_numpy()
        # cbg_map = {str(x[0])[:-3]: x[1] for x in cbg_map}
        # gdf['CDISTRICT'] = list(map(lambda x: cbg_map[x], gdf['GEOID20']))
        # gdf = gdf[[x in {4,5} for x in gdf["CDISTRICT"]]]  # 1,4,5
        # dissolve into higher level
        # gdf = gdf.dissolve(by="TRACT", aggfunc="sum")
        gdf = gdf.dissolve(by="COUNTY", aggfunc="sum")
        gdf = gdf.reset_index()
        # gdf = gdf.explode(ignore_index=True)
        # gdf = gdf.groupby(by='TRACT').first()
        shapefile = root+"_proc.shp"
        gdf.to_file(shapefile, encoding='UTF-8')
        g_data = adjacency_from_shp(shapefile)
        # Extra bs for sample
        # g_data[12].append(13)
        # g_data[12].remove(22)
        # g_data[13].append(12)
        # g_data[22].remove(12)
        loc_df = gpd.read_file(shapefile, encoding='UTF-8')
        # loc_df['centroid_column'] = loc_df.centroid
        # centers = loc_df.set_geometry('centroid_column')
        # centers.set_index('OBJECTID', inplace=True)
        # print(centers)
        # y_locs = {x: centers.loc[x]['centroid_column'].y for x in h.nodes}
        # stddev = np.std(np.asanyarray(list(y_locs.values())))
        # center = np.mean(
        #     np.asanyarray([y_locs[exit_edge[0]], y_locs[exit_edge[1]], y_locs[start_edge[0]], y_locs[start_edge[0]]]))
        # new_verts = [x for x in h.nodes if math.fabs(y_locs[x] - center) < stddev / 2.25]
        # h2 = h.subgraph(new_verts).copy()
        # g_data = {x: [y for y in g_data[x] if y in new_verts] for x in new_verts}
        while True:
            to_remove = []
            for v, neighbs in g_data.items():
                if len(neighbs) == 1:
                    to_remove.append(v)
            if len(to_remove) == 0:
                break
            print("Removing the vertices {0} because they have degree 1".format(to_remove))
            for v in to_remove:
                g_data[g_data[v][0]].remove(v) if len(g_data[v]) > 0 else ""
                g_data.pop(v)
        h = nx.DiGraph(incoming_graph_data=g_data)
        success, counterexample = nx.check_planarity(h, counterexample=True)
        while not success:
            # spring layouts look somewhat normal-- we cannot get a near planar layout, because that can only be built from
            # planar graphs
            some_layout = nx.spring_layout(counterexample)
            # nx.draw(h, pos=nx.nx_pydot.pydot_layout(counterexample, prog="dot"))
            # plt.show()
            # Use the Bentley-Ottman algorithm to find bad edges
            cross_edges = find_intersecting_edges(counterexample, some_layout)
            rev_cross_edges = list([(e[1], e[0]) for e in cross_edges])
            # Remove bad edges
            h.remove_edges_from(cross_edges + rev_cross_edges)
            for start, stop in cross_edges + rev_cross_edges:
                g_data[start].remove(stop)  # need to adjust g_data as well
            print("Removing edges {0} to make planar".format(cross_edges))
            # nx.draw(counterexample, pos=some_layout, with_labels=True)
            # plt.show()
            success, counterexample = nx.check_planarity(h, counterexample=True)

        g = counterexample
        positions = nx.planar_layout(g)
        # Remove any newly created islands or dangling edges
        # Make sure the graph only has one connected component
        cocos = list(nx.connected_components(g))
        if len(cocos) > 1:  # just remove the islands
            max_ind = np.argmax([len(coco) for coco in cocos])
            g = generate_face_order.make_subgraph(g, cocos[max_ind])  # Always the biggest
        # graph is good: save it if it hasn't been saved
        if not os.path.exists(root + ".adjlist"):
            nx.write_adjlist(g, root+".adjlist")
        print("Wrote calculated adjacency to {0}".format(root + ".adjlist"))
        # g = nx.PlanarEmbedding()

    # Sort angles w/o computing atan2, slightly faster for a computationally insignificant portion:
    # Input:  dx, dy: coordinates of a (difference) vector.
    # Output: a number from the range [-2 .. 2] which is monotonic
    #         in the angle this vector makes against the x axis.
    #         and with the same discontinuity as atan2
    def pseudoangle(x, vect2):
        vect1 = np.array(positions[x])
        vect = vect1 - vect2
        dx, dy = vect
        p = dx / (abs(dx) + abs(dy))  # -1 .. 1 increasing with x
        if dy < 0:
            return p - 1  # -2 .. 0 increasing with x
        else:
            return 1 - p  # 0 .. 2 decreasing with x

    # oriented_g_data = {}
    # for v, neighbs in g_data.items():
    #     # Sort neighbors by orientation of vectors
    #     # vect2 = np.array([centers.loc[v]['centroid_column'].x,
    #     #                   centers.loc[v]['centroid_column'].y]).flatten()
    #     vect2 = np.array(positions[v])
    #     new_neighb = sorted(neighbs, key=lambda x: pseudoangle(x, vect2), reverse=True)
    #     # print("{0}: {1}".format(v, new_neighb))
    #     oriented_g_data[v] = new_neighb
    # g.set_data(oriented_g_data)
    success, counterexample = nx.check_planarity(g, counterexample=True)
    if not success:
        nx.draw(counterexample, pos=positions, with_labels=True)
        plt.show()
        print("Error: Adjacency graph is not planar, exiting...")
        exit(0)
    # g.check_structure()

    # start the algorithm!
    # exit_edge = (308, 306)
    # start_edge = (308, 306)
    # exit_edge = (404, 403)
    # start_edge = (404, 403)
    exit_edge = (6, 48)
    start_edge = (14, 37)
    # exit_edge = (11,12)
    # start_edge = (12,13)
    # outer_face = max([g.traverse_face(*exit_edge), g.traverse_face(exit_edge[1], exit_edge[0])],
    #                  key=lambda x: len(x))
    cts = []
    efficiencies = []
    pops = []
    # print("Sampling with start edge {0} and exit edge {1}".format(start_edge, exit_edge))
    k = 2
    num_samples = 100
    cont_sections, count, sample_paths, outer_boundary, h2, face_order = count_and_sample(draw, face_order, g, positions, exit_edge, start_edge, k, num_samples, root, recalculate)
    if len(sample_paths[-1]) == 0:
        raise Exception("None of the sampled paths survived.")
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    for path in sample_paths:
        ct, comps, edges, new_map = eval_path(path, cont_sections, copy.deepcopy(h2), positions, face_order, outer_boundary, k, loc_df)
        if len(comps) != k:
            continue
        sums = [0,]*k
        for i in range(len(comps)):
            for v in comps[i]:
                sums[i] += loc_df.loc[v]['POP100']
        # Population count is unrealistic for exploded graphs, so ignore?
        if np.max(sums) - np.min(sums) > 50000 or ct > 65:
            # print("Refusing a population {1} standard deviation of {0}".format(np.std(sums), sums))
            continue
        print("Found a good partition {0} with std {1}".format(sums, np.std(sums)))
        draw_maps(comps, edges, new_map, loc_df, positions, draw3=True)
        # sum1 = 0
        # sum2 = 0
        # efficiencies.append(calculate_eff_gap(g1, g2, loc_df, sum1, sum2))
        # print("{0}, {1}".format(sum1, sum2))
        # efficiencies.append(calculate_eff_gap(g1, g2, loc_df, sum1, sum2))
        cts.append(ct)
        pops.append(sums)
    # print("Found {0} possible partitions".format(len(efficiencies)))
    print("Path length distribution: Mean {0} with variance {1}, shortest path {2}".format(np.mean(cts), np.std(cts),
                                                                                           np.min(cts)))
    # print("Diversity distribution: Sampled {2} with mean {0} with variance {1}".format(np.mean(diversities), np.std(diversities), len(diversities)))
    # print("Efficiency gap distribution: Sampled {2} with mean {0} with variance {1}".format(np.mean(efficiencies),
    #                                                                                         np.std(efficiencies),
    #                                                                                         len(efficiencies)))
    print("Path lengths: {0}\n Population gaps: {1}".format(cts, pops))
    # sum1 = 0
    # sum2 = 0
    # g1 = []
    # g2 = []
    # for v in h2.nodes:
    #     if loc_df.loc[v]['DISTRICT'] == '27':
    #         # sum1 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
    #         sum1 += loc_df.loc[v]['PERSONS']
    #         g1.append(v)
    #     else:
    #         # sum2 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
    #         sum2 += loc_df.loc[v]['PERSONS']
    #         g2.append(v)
    # print("True diversity was: " + str(math.fabs(sum2 - sum1)))
    # print("True efficiency gap was: " + str(calculate_eff_gap(g1, g2, loc_df, sum1, sum2)))
    # print("True pop gap was: " + str(abs(sum2 - sum1)))
    print("Finish time: " + str(time.time()))
    return count


def count_and_sample(draw, face_order, g, positions, exit_edge, start_edge, num_distr, num_samples, root, recalculate):
    # exit_edge = (71, 74)
    # start_edge = (46, 48)
    outer_face_edge = exit_edge  # The edge where the outer face is cut
    geom_dict = {key: generate_face_order.Point(positions[key]) for key in positions}

    # Order the faces according to face_order
    outer_face = min([g.traverse_face(*outer_face_edge), g.traverse_face(outer_face_edge[1], outer_face_edge[0])],
                     key=lambda x: len(x) if all(generate_face_order.vert_in_face(g.nodes, x, positions, geom_dict)) else np.infty)
    # outer_face.append(outer_face_edge)
    start_boundary_list = []
    start_boundary_labels = []
    for i in range(len(outer_face)):
        # edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
        edge = (outer_face[i], outer_face[(i + 1) % len(outer_face)])
        # if edge == exit_edge or edge == (exit_edge[1], exit_edge[0]):
        # if edge == outer_face_edge or edge == (outer_face_edge[1], outer_face_edge[0]):
        #     continue
        start_boundary_list.append(edge)
        start_boundary_labels.append(1 if edge == start_edge or edge == exit_edge else 0)
    # First enumerate all faces
    face_dict = {}
    # Clean bad/unnecessary parts of the graph
    start_boundary_list, h2 = clean_graph(exit_edge, face_dict, g, positions, start_boundary_labels,
                                          start_boundary_list, start_edge, geom_dict)
    if draw:
        # ax = plt.subplot(121)
        # plt.sca(ax)
        plt.figure(figsize=(65, 65))
        nx.draw(h2, pos=positions, node_size=30, with_labels=True, font_size=8, linewidths=0,
                width=.2)
        # frontiers = [[41, 15, 10], [59, 37, 15, 10], [75, 71, 59, 37, 15, 10], [121, 124, 72, 30, 2, 10], [131, 124, 72, 30, 2, 10], [130, 129, 30, 2, 10], [128, 30, 2, 10], [137, 30, 2, 10], [143, 141, 137, 30, 2, 10], [142, 141, 137, 30, 2, 10], [147, 141, 137, 30, 2, 10], [146, 136, 137, 30, 2, 10], [94, 93, 136, 137, 30, 2, 10], [39, 38, 46, 136, 137, 30, 2, 10], [36, 38, 46, 136, 137, 30, 2, 10], [14, 38, 46, 136, 137, 30, 2, 10], [14, 38, 46, 136, 137, 30, 2, 1, 816], [14, 38, 46, 136, 137, 30, 2, 1, 816, 707, 704], [14, 38, 46, 136, 137, 30, 2, 1, 816, 707, 705, 703], [14, 11, 661, 663, 664, 667, 670, 690, 689, 686, 677, 674], [151, 486, 284, 473, 495, 496, 782, 687, 686, 677, 674], [151, 486, 284, 473, 495, 496, 782, 687, 686, 676], [152, 66, 151, 486, 284, 473, 495, 496, 782, 687, 686, 676], [127, 66, 151, 486, 284, 473, 495, 496, 782, 687, 686, 676], [127, 66, 151, 486, 284, 473, 495, 496, 782, 687, 686], [126, 70, 66, 151, 486, 284, 473, 495, 496, 782, 687, 686], [115, 60, 135, 139, 291, 284, 473, 495, 496, 782, 687, 686], [132, 135, 139, 291, 284, 473, 495, 496, 782, 687, 686], [132, 135, 139, 291, 284, 473, 495, 496, 782, 781], [132, 135, 139, 291, 284, 473, 495, 496, 783], [132, 135, 139, 291, 284, 473, 784], [132, 135, 139, 291, 284, 473, 785], [132, 135, 139, 291, 284, 283, 810], [132, 135, 139, 291, 288, 282, 320, 807], [132, 135, 139, 291, 288, 282, 320, 806], [132, 135, 139, 291, 288, 282, 320, 319], [132, 135, 139, 291, 288, 282, 320, 262, 263, 253], [132, 135, 140, 298, 244, 252], [132, 135, 140, 298, 306, 310], [132, 135, 759, 756, 744, 736], [132, 135, 759, 756, 744, 743, 742], [132, 135, 759, 756, 788, 792, 793], [132, 135, 759, 756, 788, 797], [132, 135, 759, 768]]
        # edge_bunch = []
        # edge_cmap = []
        # for i in range(len(frontiers)):
        #     frontier = frontiers[i]
        #     edge_bunch += list([(frontier[f], frontier[f+1]) for f in range(len(frontier)-1)])
        #     edge_cmap += [i,]*(len(frontier)-1)
        # nx.draw_networkx_edges(h2, pos=positions, edgelist=edge_bunch, edge_color=edge_cmap, edge_cmap=plt.cm.plasma)
        plt.show()
        # G2 = h.subgraph([106,0,3,4,5,6,7,8,9,14,107,13,51,52])
        # nx.draw(G2, pos=positions, with_labels=True)
        # centers.plot()
        # loc_df.plot()
        plt.savefig(root)
        exit()


    if not os.path.exists(root + ".order") or recalculate:
        outer_face = min([h2.traverse_face(*start_edge), h2.traverse_face(start_edge[1], start_edge[0])],
                         key=lambda x: len(x) if all(generate_face_order.vert_in_face(h2.nodes, x, positions, geom_dict)) else np.infty)
        outer_face_edges = {(outer_face[i], outer_face[(i + 1) % len(outer_face)]) for i in range(len(outer_face))}
        rev_outer_face_edges = {(outer_face[(i + 1) % len(outer_face)], outer_face[i]) for i in range(len(outer_face))}
        # face_order = generate_face_order.order_faces(h2, start_edge, positions, geom_dict, outer_face_edges.union(rev_outer_face_edges))
        face_order = generate_face_order.order_faces(h2, start_edge, positions, geom_dict, {start_edge, (start_edge[1], start_edge[0])})
        with open(root + ".order", "w") as out_file:
            for f in face_order:
                for x in f:
                    out_file.write(str(x) + ',')
                out_file.write("\n")
    else:
        with open(root + ".order") as in_file:
            face_order = in_file.readlines()
            face_order = list([list([int(y) for y in x[:-2].split(',')]) for x in face_order])

    # set start_edge and end_edge according to the traversal
    # start_edge = tuple(x for x in face_order[0] if x in outer_face)
    # exit_edge = tuple(x for x in face_order[-1] if x in outer_face)
    # if not generate_face_order.same_face(h2.traverse_face(*start_edge), outer_face):
    #     start_edge = (start_edge[1], start_edge[0])
    # if not generate_face_order.same_face(h2.traverse_face(*exit_edge), outer_face):
    #     exit_edge = (exit_edge[1], exit_edge[0])
    face_order = orient_faces(face_order, h2, positions, start_edge)
    # Test that face_order faces actually exist
    # for face in face_order:
    #     oface = ensure_cw(face, positions)
    #     sedges = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
    #     oedges = [(oface[i], oface[(i + 1) % len(oface)]) for i in range(len(oface))]
    #     for f in sedges:
    #         if f not in oedges:
    #             print("This face is oriented incorrectly: " + str(face) + ", not like " + str(oface))
    #             break
    to_rem = []
    for i in range(len(face_order)):
        face = face_order[i]
        f_set = set(face)
        d_sets = [set(f) for f in face_dict.values()]
        if f_set not in d_sets:
            print("This face was in face_order but not in dict: " + str(face))
            to_rem.append(i)
        if np.count_nonzero([generate_face_order.same_face(y, face) for y in face_order]) > 1:
            prev_inds = [y for y in to_rem if generate_face_order.same_face(face_order[y], face)]
            if len(prev_inds) > 0:
                to_rem.remove(prev_inds[0])
            to_rem.append(i)
    to_rem = [face_order[i] for i in to_rem]
    for face in to_rem:
        face_order.remove(face)
    for face in face_dict.values():
        f_set = set(face)
        d_sets = [set(f) for f in face_order]
        if f_set not in d_sets:
            print("This face was in dict but not in face_order: " + str(face))
            for f_ind in range(len(face_order)):
                if any([x in face for x in face_order[f_ind]]):
                    face_order = face_order[:f_ind+1] + [face] + face_order[f_ind+1:]
                    break
    # Code to print out adjacency matrix for online viewer:
    # cur_verts = []
    # for v in face_dict.values():
    #     cur_verts += v
    # s_verts = sorted([v for v in g.nodes if v in cur_verts])
    # mat = nx.adjacency_matrix(g, nodelist=s_verts).toarray()
    # for i in range(len(mat)):
    #     row = mat[i]
    #     print(s_verts[i], end=': ')
    #     for x in row:
    #         print(str(int(x/2))+', ', end='')
    #     print()
    # print(face_dict)
    # exit()
    # Then sort faces by lexicographic y coordinates
    # face_list = sorted(face_dict.values(), key=lambda face: np.mean([positions[x][0] for x in face]))
    cont_sections, face_list = create_face_order(start_edge, face_order, positions, start_boundary_list, g, geom_dict)
    # Successfully created face_list!
    # exit()
    # Code to measure space needed (depends on face_list):
    # nth Motzkin number https://oeis.org/A001006
    m_n = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467, 2356779, 6536382,
           18199284, 50852019, 142547559, 400763223, 1129760415, 3192727797, 9043402501, 25669818476, 73007772802,
           208023278209, 593742784829]
    total_mem = 0
    c = 0
    for sections in cont_sections:
        sect_mem = 1
        for sect in sections:
            # sect_mem *= m_n[len(sect)] if len(sect) < len(m_n) else m_n[-1]
            sect_mem *= sum([1 / (d2 + 1) * (math.comb(2 * d2, d2) * math.comb(len(sect), 2 * d2)) for d2 in
                             range(depth_bound + 1)])
        print("{0}: {1}: {2}".format(c, '.'.join([str(len(sect)) for sect in sections]), sect_mem))
        c += 1
        total_mem += sect_mem
    print("Will be using approximately {0} entries.".format(total_mem))
    print("The maximum length kappa of a frontier is {0}".format(max([len(x[0]) for x in cont_sections if len(x) > 0])))
    # exit(0)
    print(face_list)
    print("Starting table allocation and edge map creation.")

    cont_sections = [[[]]] + cont_sections  # add the initial empty section

    db_name = root + '_sql.db'
    if not os.path.exists(db_name):
        fd = open(db_name, "x")
        fd.close()
    conn = sqlite3.connect(db_name)
    # conn = None
    table, edge_maps = allocate_table(face_list, start_boundary_list, cont_sections, num_distr, conn=conn, just_sample=True)
    # np.save('saved_table', table)
    # np.save('saved_table', table)
    print("Finished setup: " + str(time.time()))
    sample_paths, count = count_non_int_paths_w_table(table, edge_maps, num_distr, num_samples, conn=conn, just_sample=True)
    conn.close() if conn is not None else ""
    trimmed_sample_paths = list([p[0] for p in sample_paths])
    # for path in sample_paths:
    #     if path[1] == 2*(num_distr-1)-1:
    #         trimmed_sample_paths.append(path[0])
    # count, sample_paths = count_non_int_paths(face_list, start_boundary_list, cont_sections, num_distr)
    print("Counted " + str(count) + " non-self-intersecting paths")
    return cont_sections, count, trimmed_sample_paths, start_boundary_list, h2, face_list


def create_face_order(start_edge, face_order, positions, start_boundary_list, g, geom_dict):
    # Ensure that face_list results in a continuous boundary
    # Iterate through face_list keeping track of contiguous boundary sets
    # Store the contiguous sections as a list (each frontier) of lists (each connected component) of lists (frontiers)
    cont_sections = []
    face_list = []
    cur_boundary = copy.deepcopy(start_boundary_list)
    prev_ver = [start_edge]
    # A solution to incorrect face traversals is using a wait queue. This takes impossible faces, adds them to a queue,
    # and inserts them whenever they become possible
    # [238, 239, 316, 319, 242]
    wait_queue = []
    face_order_index = 0
    while face_order_index < len(face_order) or len(wait_queue) > 0:
        if len(cont_sections) > 0:
            sect_boundary = [x[0] for x in cont_sections[-1][0]] + [cont_sections[-1][0][-1][1]]
            if any(sect_boundary.count(x) > 1 for x in sect_boundary):
                raise BaseException("Boundary got messed up.")
            for x in range(2, len(sect_boundary)):
                for offset in range(2, min(x, 10)):
                    if sect_boundary[x-offset] in g[sect_boundary[x]]:
                        to_add1 = []
                        for face in wait_queue:
                            if all(generate_face_order.vert_in_face(face, sect_boundary[x-offset:x+1], positions, geom_dict)):
                                wait_queue.remove(face)
                                to_add1.append(face)
                        to_add2 = []
                        for face in face_order[face_order_index:]:
                            if all(generate_face_order.vert_in_face(face, sect_boundary[x-offset:x+1], positions, geom_dict)):
                                if face not in wait_queue:
                                    face_order.remove(face)
                                    to_add2.append(face)
                        wait_queue = to_add1 + to_add2 + wait_queue  # add to the beginning of wait_q
                        # face = [f for f in face_order if generate_face_order.same_face(f, sect_boundary[x-2:x+1])]
                        # if len(face) == 0:  # we have other faces in between:
            print("Boundary length: {0}\t Wait queue length: {1}".format(len(sect_boundary), len(wait_queue)))
        # if len(cont_sections) > 0 and cont_sections[-1] == [[(110, 38), (38, 170), (170, 32), (32, 112)]]:
        #     print("")
        # Set up vertices in boundary
        boundary_verts = [x[0] for x in cur_boundary] + [cur_boundary[-1][1]]
        face = face_order[face_order_index] if face_order_index < len(face_order) else None
        face_order_index += 1
        for q_face in wait_queue:
            # Check if face can now be added
            if any([(q_face[x], q_face[(x+1) % len(q_face)]) in cur_boundary for x in range(len(q_face))]) and\
                    np.count_nonzero([any([q_face[x] in e for x in range(len(q_face))]) for e in cur_boundary]) - \
                    np.count_nonzero([(q_face[x], q_face[(x + 1) % len(q_face)]) in cur_boundary for x in range(len(q_face))]) <= 2:
                face = q_face
                face_order_index -= 1
                wait_queue.remove(q_face)
                break
            elif np.count_nonzero([[any([q_face[x] in e for x in range(len(q_face))]) for e in cur_boundary]]) == 0 and\
                not generate_face_order.vert_in_face(q_face[0], boundary_verts, positions, geom_dict):
                # Some faces are filled in from things around them-- this means everything is okay, just remove it from the queue
                wait_queue.remove(q_face)
        if face is None:
            raise BaseException("Failed to generate a valid traversal.")
        print(face) if debug else ""
        print(cur_boundary) if debug else ""
        if not any((face[x], face[(x+1) % len(face)]) in cur_boundary for x in range(len(face))):
            # Algorithm messed up, add to queue-- the second condition checks for closed loops
            wait_queue.append(face)
            continue
        elif np.count_nonzero([any([face[x] in e for x in range(len(face))]) for e in cur_boundary]) - \
                np.count_nonzero([(face[x], face[(x+1) % len(face)]) in cur_boundary for x in range(len(face))]) > 2:
            # catch cases that use the outer boundary
            # if np.count_nonzero([any([face[x] in e for x in range(len(face))]) for e in prev_ver]) - \
            #         np.count_nonzero([(face[x], face[(x+1) % len(face)]) in prev_ver for x in range(len(face))]) <= 2:
            #     pass
            # else:
            wait_queue.append(face)
            continue
        face = list(reversed(face))
        # Make sure orientation of face is good
        m_ind = 0  # Index of the maximum vertex in the face that leaves the current boundary
        for i in range(len(face)):
            if face[i] in boundary_verts:
                m_ind = boundary_verts.index(face[i]) if boundary_verts.index(face[i]) >= m_ind else m_ind
        start_ind = face.index(boundary_verts[m_ind])
        # Rotate current face to have m_ind be first
        face = [face[(x + start_ind) % len(face)] for x in range(len(face))]
        # stores the new vertices that will be added to the boundary
        new_loc = []
        # stores the index all face elements will be put to
        index = len(cur_boundary)
        for i in range(len(face)):
            edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed?
            if edge in cur_boundary:
                cur_index = cur_boundary.index(edge)
                cur_boundary.pop(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                # Can't just do this, want the original order and need cont_section to just make a new section
                new_loc.append((edge[1], edge[0]))
        if len(set(cur_boundary).intersection(set(start_boundary_list))) == 0 and face_order_index != len(face_order):  # We circled inside!
            # flip the order of the rest of face_order and reset the problem
            face_order_index -= 1
            face_order = face_order[:face_order_index] + list(reversed(face_order[face_order_index:]))
            cur_boundary = [(boundary_verts[z], boundary_verts[z+1]) for z in range(len(boundary_verts)-1)]
            continue
        # Make sure new_loc follows the order of boundary:
        # swapped = True
        # swapped = len(cur_boundary) > 0 and len(new_loc) > 0 and \
        #           (len(cur_boundary) > index > 0 and (new_loc[0][0] not in cur_boundary[index - 1] or
        #                                               new_loc[-1][1] not in cur_boundary[index]))
        # swapped = len(cur_boundary) > 0 and len(new_loc) > 0 and \
        #           (len(cur_boundary) > index > 0 and (
        #                   len(set(new_loc[0]).intersection(set(cur_boundary[index - 1]))) == 0 or len(
        #               set(new_loc[-1]).intersection(set(cur_boundary[index])))) == 0)
        if len(new_loc) > 1:  # need to find where to start the new locations
            start_ind = 0
            for i in range(len(new_loc)):
                this_loc = set(new_loc[i])
                prev_loc = set(new_loc[(i - 1) % len(new_loc)])
                start_spot = set(cur_boundary[index - 1]) if index > 0 else None
                end_spot = set(cur_boundary[index]) if index < len(cur_boundary) else None
                # Put prev_loc and this_loc between start_spot and end_spot if it fits
                if start_spot is not None and len(this_loc.intersection(start_spot)) > 0:
                    start_ind = i
                    if len(this_loc.intersection(start_spot)) == 1 and end_spot is not None and len(prev_loc.intersection(end_spot)) == 1:
                        print("Found good rotation: " + str(new_loc)) if debug else ""
                        break
                    elif end_spot is None and len(this_loc.intersection(start_spot)) == 1:
                        break
                elif start_spot is None:
                    start_ind = i
                    break
            new_loc = [new_loc[(x + start_ind) % len(new_loc)] for x in range(len(new_loc))]
        # Find which parts of cont_sections new_loc matches with
        # Need to merge sections/ remove them
        prev_ver = copy.deepcopy(cont_sections[-1]) if len(cont_sections) > 0 else []
        # for exit_edge in exits:
        #     # Special case for different exit_edges, make sure it's added to new_loc
        #     if exit_edge[0] in face and exit_edge[1] in face:
        #         new_loc.append(exit_edge)
        #     # special case for strange exit_edges
        #     if exit_edge in new_loc or (exit_edge[1], exit_edge[0]) in new_loc:
        #         prev_ver.append([(exit_edge[1], exit_edge[0])])
        #         try:
        #             new_loc.remove((exit_edge[1], exit_edge[0]))
        #         except ValueError:
        #             new_loc.remove(exit_edge)

        if len(cont_sections) == 0:  # special first-time setup
            cont_sections.append([new_loc])
        elif len(new_loc) == 0:  # Another special case, just remove appropriate edges
            if face_order_index == len(face_order) and len(wait_queue) == 0:
                # for some reason suddenly failed, just fix last step
                cont_sections.append([[]])
                face = ensure_ccw(face, positions)
                face_list.append(face)
                break
            offset = 0
            for j in range(len(prev_ver)):
                for i in range(len(face)):
                    edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
                    if edge in prev_ver[j - offset]:
                        prev_ver[j - offset].remove(edge)
                        if len(prev_ver[j - offset]) == 0:
                            prev_ver.pop(j - offset)
                            offset += 1
            cont_sections.append(prev_ver)
        else:
            new_section = True
            cont_section_verts = [[edge[0] for edge in section] + [section[-1][1]] for section in cont_sections[-1]]
            # new_loc_verts = [edge[0] for edge in new_loc] + [new_loc[-1][1]]
            for i in range(len(cont_section_verts)):
                section_vert = cont_section_verts[i]
                truth_table = (new_loc[0][0] in section_vert, new_loc[-1][1] in section_vert)
                if truth_table == (True, True):  # Just add to current cont_section
                    cont_sections += [prev_ver]
                    cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc \
                                           + prev_ver[i][section_vert.index(new_loc[-1][1]):]
                    new_section = False
                    break
                elif truth_table == (False, False):
                    continue
                elif truth_table == (True, False):  # Need to connect two sections to create a new one - or extend!
                    should_extend = True
                    for k in range(len(cont_section_verts)):
                        if new_loc[-1][1] in cont_section_verts[(i + k) % len(cont_section_verts)]:
                            prev_ver[i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
                                          prev_ver[(i + k) % len(prev_ver)][
                                          cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]
                            prev_ver.pop((i + k) % len(prev_ver))
                            # cont_sections += [prev_ver[:i]] if i + k != len(cont_section_verts) else [prev_ver[1:i]]
                            # cont_sections[-1] += [prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
                            #                       prev_ver[(i + k) % len(prev_ver)][
                            #                       cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]]
                            # cont_sections[-1] += prev_ver[i + k + 1:]
                            cont_sections += [prev_ver]
                            should_extend = False
                    if should_extend:  # Extend!
                        cont_sections += [prev_ver]
                        cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc
                    new_section = False
                    break
                elif truth_table == (False, True):  # Extend at the beginning- only if new_loc[0][0] nowhere else
                    should_continue = False
                    for k in range(len(cont_section_verts)):
                        if new_loc[0][0] in cont_section_verts[k]:
                            should_continue = True
                    if should_continue:  # Very awkward way to skip
                        continue
                    # have to watch out for when we are cutting on the boundary!
                    # if new_loc[0][0] in
                    cont_sections += [prev_ver]
                    cont_sections[-1][i] = new_loc + prev_ver[i][section_vert.index(new_loc[-1][1]):]
                    new_section = False
                    break
            if new_section:
                prev_ver.append(new_loc)
                cont_sections.append(prev_ver)
        print([[edge[0] for edge in section] + [section[-1][1]] for section in cont_sections[-1]]) if debug else ""
        # Perform additional check to connect cont_sections we might have missed
        # This can happen if a new face connects to multiple cont_sections, and we just add it to the first one
        # for i in range(len(cont_sections[-1])):
        #     # prev_ver[i]'s last vertex == prev_ver[i+1]'s first
        #     if cont_sections[-1][i][-1][1] == cont_sections[-1][(i + 1) % len(cont_sections[-1])][0][0]:
        #         # Connect!
        #         cont_sections[-1][i] += cont_sections[-1][(i + 1) % len(cont_sections[-1])]
        #         cont_sections[-1].pop((i + 1) % len(cont_sections[-1]))
        #         break
        cur_boundary = cur_boundary[:index] + new_loc + cur_boundary[index:]
        face = ensure_ccw(face, positions)
        face_list.append(face)
    return cont_sections, face_list


def clean_graph(exit_edge, face_dict, g: nx.PlanarEmbedding, positions, start_boundary_labels, start_boundary_list, start_edge, geom_dict):
    # Some faces have self loops, we can remove the inner loops and all vertices within
    verts_to_clean = set()
    points_to_keep = set()
    for edge in g.edges:
        # sorted_edge = sorted(edge, key=lambda x: positions[x][1])  # unnecessary to do this I think
        face = g.traverse_face(*edge)  # Traverse clockwise
        # if all([generate_face_order.vert_in_face(x, face, positions, geom_dict) for x in g.nodes]):  # Hardcode outer face
        if len(face) > 25:
            continue
        face = ensure_ccw(face, positions)
        if len(np.unique(face)) != len(face):  # bad face
            point_ind = np.argmax([face.count(x) for x in face])
            point = face[point_ind]
            i1 = face.index(point)  # First occurence
            i2 = face.index(point, i1 + 1)  # Second
            f1 = face[i1:i2]  # option 1 for bad face
            f2 = face[i2:] + face[:i1]  # option 2
            bad_face, good_face = (f1, f2) if all(generate_face_order.vert_in_face(f1, f2, positions, geom_dict)) else (f2, f1)
            verts_to_clean = verts_to_clean.union(set(bad_face[1:]))
            points_to_keep.add(point)
            face_dict[str(sorted(good_face))] = good_face
        else:
            face_dict[str(sorted(face))] = face
    # Remove boundary loops
    while len(np.unique(np.asanyarray(start_boundary_list).flatten())) * 2 != len(
            np.asanyarray(start_boundary_list).flatten()):
        flat_boundary = np.asanyarray(start_boundary_list).flatten()
        unique, indices, counts = np.unique(flat_boundary, return_counts=True, return_index=True)
        point_ind = np.argmax(counts)
        point = unique[point_ind]
        i1 = indices[point_ind]
        i2 = np.where(flat_boundary == point)[0][2]
        verts_to_clean = verts_to_clean.union(set(flat_boundary[i1 + 1:i2]))
        points_to_keep.add(point)
        start_boundary_list = start_boundary_list[:int(np.floor(i1 / 2) + 1)] + start_boundary_list[
                                                                                int(np.floor(i2 / 2) + 1):]
        start_boundary_labels = start_boundary_labels[:int(np.floor(i1 / 2) + 1)] + start_boundary_labels[
                                                                                    int(np.floor(i2 / 2) + 1):]
    # Cut off ears with no entry:
    flat_boundary = [edge[0] for edge in start_boundary_list] + [start_boundary_list[-1][1]]
    for e in g.edges:
        if e[0] in flat_boundary and e[1] in flat_boundary:
            i1 = flat_boundary.index(e[0])
            i2 = flat_boundary.index(e[1])
            if (i2 - i1) % len(flat_boundary) <= 2 or (i1 - i2) % len(flat_boundary) <= 2:
                continue
            if i1 < i2:
                inside_slice = flat_boundary[i1 + 1:i2]
            else:
                inside_slice = flat_boundary[:i2] + flat_boundary[i1 + 1:]
            if (start_edge[0] not in inside_slice and start_edge[1] not in inside_slice) \
                    and (exit_edge[0] not in inside_slice and exit_edge[1] not in inside_slice):
                # Cut out everything within, path may never enter
                points_to_keep.add(e[0])
                points_to_keep.add(e[1])
                verts_to_clean = verts_to_clean.union(set(inside_slice))
                if i1 < i2:
                    start_boundary_list = start_boundary_list[:i1] + [(e[0], e[1])] + start_boundary_list[i2:]
                    start_boundary_labels = start_boundary_labels[:i1] + [0] + start_boundary_labels[i2:]
                else:
                    start_boundary_list = start_boundary_list[i2:i1 + 1]
                    start_boundary_labels = start_boundary_labels[i2:i1 + 1]
                # reset flat_boundary
                flat_boundary = [edge[0] for edge in start_boundary_list] + [start_boundary_list[-1][1]]
    # Deal with bad faces
    while True:
        to_rem = []
        found = False
        for f_name, f in face_dict.items():
            if len(set(f).intersection(verts_to_clean)) != 0:
                found = True
                to_rem.append(f_name)
                verts_to_clean = verts_to_clean.union(set(f)).difference(points_to_keep)
        print("Removing: " + str(to_rem))
        for f_name in to_rem:
            face_dict.pop(f_name)
        if not found:
            break
    verts_left = set()
    for f in face_dict.values():
        verts_left = verts_left.union(set(f))
    # verts_left = verts_left.union()
    h2 = generate_face_order.make_subgraph(g, list(verts_left))
    # h2 = g.subgraph(list(verts_left)).copy()
    return start_boundary_list, h2


def calculate_eff_gap(g1, g2, loc_df, sum1, sum2):
    repct1 = 0
    repct2 = 0
    demct1 = 0
    demct2 = 0
    for v in g1:
        # sum1 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
        repct1 += loc_df.loc[v]['Rep12']
        demct1 += loc_df.loc[v]['Dem12']
    for v in g2:
        # sum2 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
        repct2 += loc_df.loc[v]['Rep12']
        demct2 += loc_df.loc[v]['Dem12']
    # diversities.append(math.fabs(sum1 - sum2))
    # Compute efficiency gap
    print("Reps won both" if demct1 < repct1 and demct2 < repct2 else "Dems won at least one")
    dem_waste1 = demct1 if demct1 < repct1 else (repct1 - demct1) / 2
    rep_waste1 = repct1 if repct1 < demct1 else (demct1 - repct1) / 2
    dem_waste2 = demct2 if demct2 < repct2 else (repct2 - demct2) / 2
    rep_waste2 = repct2 if repct2 < demct2 else (demct2 - repct2) / 2
    return ((dem_waste1 - rep_waste1) / sum1 + (dem_waste2 - rep_waste2) / sum2) / 2


def eval_path(path, cont_sections, g, positions, face_list, outer_face, k, loc_df, draw2=False, draw3=False):
    edges = set()
    outer_edges = set(outer_face)
    prev_ones = 0
    for i in range(1, len(path)):
        assignment = path[i]
        cur_ones = assignment.count('1')
        sect = cont_sections[i]
        flat_sect = [x for j in range(len(sect)) for x in sect[j]]  # only for multi sections
        if abs(cur_ones - prev_ones) == 1:
            face = face_list[i-1]
            face_edges = {tuple(sorted([face[z], face[(z+1) % len(face)]])) for z in range(len(face))}
            # We have an exit edge on the outside boundary of the face corresponding to this step
            exits = face_edges.intersection(outer_edges)
            if len(exits) > 0:  # only take one arbitrary exit per outer face
                exits = set([exits.pop()])
            edges = edges.union(exits)
            # edges = edges.union({(e[1], e[0]) for e in exits})
        edges = edges.union(set([flat_sect[i] for i in range(len(assignment)) if assignment[i] != '0']))
        # edges = edges.union(
        #     set([(flat_sect[i][1], flat_sect[i][0]) for i in range(len(assignment)) if assignment[i] != '0']))
        prev_ones = cur_ones
    g = generate_face_order.remove_planar_edges(g, edges)
    # g.remove_edges_from(edges)
    comps = list(nx.connected_components(g))
    draw_maps(comps, edges, g, loc_df, positions, draw2, draw3)
    if len(comps) != k:
        print(path) if debug else ""
        print("Produced {0} components".format(len(comps))) if debug else ""
        # return len(edges), "", ""
    return len(edges), comps, edges, g


def draw_maps(comps, edges, g, loc_df, positions, draw2=False, draw3=False):
    if draw2:
        plt.figure(figsize=(18, 18))
        nx.draw(g, pos=positions, node_size=60, with_labels=True, font_size=12, font_color='red', linewidths=0,
                width=.2)
        # nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=6, font_color='red')
        plt.show()
    if draw3:
        districts = [[len(comps) - i for i in range(len(comps)) if x in comps[i]] for x in loc_df.index]
        loc_df['NEW_DISTRICT'] = [x[0] if len(x) > 0 else 0 for x in districts]
        # Works for 2:
        # loc_df['NEW_DISTRICT'] = list(map(lambda x: 2 if x in comps[0] else (1 if x in comps[1] else 0), loc_df.index))
        # print(loc_df['NEW_DISTRICT'])
        fig, ax = plt.subplots()
        loc_df.plot(column='NEW_DISTRICT', ax=ax, cmap="viridis")
        plt.show()


def order_faces(graph, positions, start_edge, exit_edge):
    # Construct boundaries
    outer_face = max([graph.traverse_face(*exit_edge), graph.traverse_face(exit_edge[1], exit_edge[0])],
                     key=lambda x: len(x))
    start_boundary_list = []
    start_boundary_labels = []
    for i in range(len(outer_face)):
        # edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
        edge = (outer_face[i], outer_face[(i + 1) % len(outer_face)])
        if edge == exit_edge or edge == (exit_edge[1], exit_edge[0]):
            continue
        start_boundary_list.append(edge)
        start_boundary_labels.append(1 if edge == start_edge else 0)
    # First enumerate all faces
    face_dict = {}
    # Some faces have self loops, we can remove the inner loops and all vertices within
    verts_to_clean = set()
    points_to_keep = set()
    for edge in graph.edges:
        sorted_edge = sorted(edge, key=lambda x: positions[x][1])  # unnecessary to do this I think
        face = graph.traverse_face(sorted_edge[1], sorted_edge[0])  # Traverse clockwise
        if len(face) > 30:  # Hardcode outer face
            continue
        face = ensure_ccw(face, positions)
        if len(np.unique(face)) != len(face):  # bad face
            point_ind = np.argmax([face.count(x) for x in face])
            point = face[point_ind]
            i1 = face.index(point)  # First occurence
            i2 = face.index(point, i1 + 1)  # Second
            verts_to_clean = verts_to_clean.union(set(face[i1 + 1:i2]))
            points_to_keep.add(point)
            face_to_add = face[:i1] + face[i2:]
            face_dict[str(sorted(face_to_add))] = face_to_add
        else:
            face_dict[str(sorted(face))] = face
    # Remove boundary loops
    while len(np.unique(np.asanyarray(start_boundary_list).flatten())) * 2 - 2 != len(
            np.asanyarray(start_boundary_list).flatten()):
        flat_boundary = np.asanyarray(start_boundary_list).flatten()
        unique, indices, counts = np.unique(flat_boundary, return_counts=True, return_index=True)
        point_ind = np.argmax(counts)
        point = unique[point_ind]
        i1 = indices[point_ind]
        i2 = np.where(flat_boundary == point)[0][2]
        verts_to_clean = verts_to_clean.union(set(flat_boundary[i1 + 1:i2]))
        points_to_keep.add(point)
        start_boundary_list = start_boundary_list[:int(np.floor(i1 / 2) + 1)] + start_boundary_list[
                                                                                int(np.floor(i2 / 2) + 1):]
        start_boundary_labels = start_boundary_labels[:int(np.floor(i1 / 2) + 1)] + start_boundary_labels[
                                                                                    int(np.floor(i2 / 2) + 1):]
    # Deal with bad faces
    while True:
        to_rem = []
        found = False
        for f_name, f in face_dict.items():
            if len(set(f).intersection(verts_to_clean)) != 0:
                found = True
                to_rem.append(f_name)
                verts_to_clean = verts_to_clean.union(set(f)).difference(points_to_keep)
        print("Removing: " + str(to_rem))
        for f_name in to_rem:
            face_dict.pop(f_name)
        if not found:
            break
    # Code to print out adjacency matrix for online viewer:
    cur_verts = []
    for v in face_dict.values():
        cur_verts += v
    s_verts = sorted([g for g in graph.nodes if g in cur_verts])
    mat = nx.adjacency_matrix(graph, nodelist=s_verts).toarray()
    for i in range(len(mat)):
        row = mat[i]
        print(s_verts[i], end=': ')
        for x in row:
            print(str(int(x / 2)) + ', ', end='')
        print()
    print(face_dict)
    exit()

    # Then sort faces by lexicographic y coordinates
    # face_list = sorted(face_dict.values(), key=lambda face: np.mean([positions[x][0] for x in face]))
    # Ensure that face_list results in a continuous boundary
    # Iterate through face_list keeping track of contiguous boundary sets
    # Store the contiguous sections as a list (each frontier) of lists (each connected component) of lists (frontiers)
    cont_sections = []
    face_list = []
    cur_edge_index = start_boundary_list.index(start_edge)
    cur_edge_index = start_boundary_list.index(
        (start_edge[1], start_edge[0])) if cur_edge_index == -1 else cur_edge_index
    cur_boundary = copy.deepcopy(start_boundary_list)
    pass_counter = 0
    shortness_param = 0
    while len(cur_boundary) > 1:
        cur_edge = cur_boundary[cur_edge_index]
        face = graph.traverse_face(cur_edge[1], cur_edge[0])
        boundary_verts = [x[0] for x in cur_boundary] + [cur_boundary[-1][1]]
        # Check if frontier still simple:
        inds = []
        for vertex in face:
            try:
                ind = boundary_verts.index(vertex)
                inds.append(ind)
            except ValueError:
                pass
        cont = False
        inds = sorted(inds)
        for i in range(len(inds) - 1):
            if inds[i + 1] - inds[i] > 1:  # Bad face! Go to next? edge
                cur_edge_index = (cur_edge_index + 3) % (len(cur_boundary) - 1) if pass_counter < len(
                    outer_face) * 20 else \
                    (cur_edge_index + 1) % (len(cur_boundary) - 1)
                cont = True
        # Also make sure boundary remains as small as possible
        if len(inds) - 1 + shortness_param <= len(face) - len(inds) + 1:  # old edges <= new edges
            if pass_counter < len(cur_boundary):
                cont = True
            else:
                shortness_param += 1
        if cont:
            pass_counter += 1
            continue
        # Face is good! Add to boundary
        pass_counter = 0
        shortness_param = 0
        # stores the new vertices that will be added to the boundary
        new_loc = []
        # stores the index all face elements will be put to
        index = len(cur_boundary)
        for i in range(len(face)):
            edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
            if edge in cur_boundary:
                cur_index = cur_boundary.index(edge)
                cur_boundary.pop(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append((edge[1], edge[0]))
        # Make sure new_loc follows the order of boundary:
        swapped = len(cur_boundary) > 0 and len(new_loc) > 0 and \
                  (len(cur_boundary) > index > 0 and (
                          len(set(new_loc[0]).intersection(set(cur_boundary[index - 1]))) == 0 or len(
                      set(new_loc[-1]).intersection(set(cur_boundary[index])))) == 0)
        if swapped:  # need to find where to start the new locations
            start_ind = -1
            for i in range(len(new_loc)):
                if len(set(new_loc[i]).intersection(set(cur_boundary[index - 1]))) > 0:
                    start_ind = i
                    if len(set(new_loc[i]).intersection(set(cur_boundary[index - 1]))) == 1 and len(
                            set(new_loc[(i - 1) % len(new_loc)]).intersection(set(cur_boundary[index]))) == 1:
                        print("Found good rotation: " + str(new_loc)) if debug else ""
                        break
            new_loc = [new_loc[(x + start_ind) % len(new_loc)] for x in range(len(new_loc))]
        # Find which parts of cont_sections new_loc matches with
        # Need to merge sections/ remove them
        if len(cont_sections) == 0:  # special first-time setup
            cont_sections.append([new_loc])
        elif len(new_loc) == 0:  # Another special case, just remove appropriate edges
            prev_ver = copy.deepcopy(cont_sections[-1])
            for j in range(len(prev_ver)):
                for i in range(len(face)):
                    edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
                    if edge in prev_ver[j]:
                        prev_ver[j].remove(edge)
                        if len(prev_ver[j]) == 0:
                            prev_ver.pop(j)
            cont_sections.append(prev_ver)
        else:
            new_section = True
            prev_ver = copy.deepcopy(cont_sections[-1])
            cont_section_verts = [[edge[0] for edge in section] + [section[-1][1]] for section in cont_sections[-1]]
            # new_loc_verts = [edge[0] for edge in new_loc] + [new_loc[-1][1]]
            for i in range(len(cont_section_verts)):
                section_vert = cont_section_verts[i]
                truth_table = (new_loc[0][0] in section_vert, new_loc[-1][1] in section_vert)
                if truth_table == (True, True):  # Just add to current cont_section
                    cont_sections += [prev_ver]
                    cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc \
                                           + prev_ver[i][section_vert.index(new_loc[-1][1]):]
                    new_section = False
                    break
                elif truth_table == (False, False):
                    continue
                elif truth_table == (True, False):  # Need to connect two sections to create a new one - or extend!
                    should_extend = True
                    for k in range(len(cont_section_verts)):
                        if new_loc[-1][1] in cont_section_verts[(i + k) % len(cont_section_verts)]:
                            prev_ver[i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
                                          prev_ver[(i + k) % len(prev_ver)][
                                          cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]
                            prev_ver.pop((i + k) % len(prev_ver))
                            # cont_sections += [prev_ver[:i]] if i + k != len(cont_section_verts) else [prev_ver[1:i]]
                            # cont_sections[-1] += [prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
                            #                       prev_ver[(i + k) % len(prev_ver)][
                            #                       cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]]
                            # cont_sections[-1] += prev_ver[i + k + 1:]
                            cont_sections += [prev_ver]
                            should_extend = False
                    if should_extend:  # Extend!
                        cont_sections += [prev_ver]
                        cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc
                    new_section = False
                    break
                elif truth_table == (False, True):  # Extend at the beginning- only if new_loc[0][0] nowhere else
                    should_continue = False
                    for k in range(len(cont_section_verts)):
                        if new_loc[0][0] in cont_section_verts[k]:
                            should_continue = True
                    if should_continue:  # Very awkward way to skip
                        continue
                    cont_sections += [prev_ver]
                    cont_sections[-1][i] = new_loc + prev_ver[i][section_vert.index(new_loc[-1][1]):]
                    new_section = False
                    break
            if new_section:
                prev_ver.append(new_loc)
                cont_sections.append(prev_ver)
        # Perform additional check to connect cont_sections we might have missed
        # This can happen if a new face connects to multiple cont_sections, and we just add it to the first one
        for i in range(len(cont_sections[-1])):
            # prev_ver[i]'s last vertex == prev_ver[i+1]'s first
            if cont_sections[-1][i][-1][1] == cont_sections[-1][(i + 1) % len(cont_sections[-1])][0][0]:
                # Connect!
                cont_sections[-1][i] += cont_sections[-1][(i + 1) % len(cont_sections[-1])]
                cont_sections[-1].pop((i + 1) % len(cont_sections[-1]))
                break
        cur_boundary = cur_boundary[:index] + new_loc + cur_boundary[index:]
        face = ensure_ccw(face, positions)
        face_list.append(face)
    # Code to measure space needed (depends on face_list):
    # nth Motzkin number https://oeis.org/A001006
    m_n = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467, 2356779, 6536382,
           18199284, 50852019, 142547559, 400763223, 1129760415, 3192727797, 9043402501, 25669818476, 73007772802,
           208023278209, 593742784829]
    total_mem = 0
    c = 0
    for sections in cont_sections:
        sect_mem = 1
        for sect in sections:
            # sect_mem *= m_n[len(sect)] if len(sect) < len(m_n) else m_n[-1]
            sect_mem *= sum([1 / (d2 + 1) * (math.comb(2 * d2, d2) * math.comb(len(sect), 2 * d2)) for d2 in
                             range(depth_bound + 1)])
        print("{0}: {1}: {2}".format(c, '.'.join([str(len(sect)) for sect in sections]), sect_mem)) if debug else ""
        c += 1
        total_mem += sect_mem
    print("Will be using approximately {0} entries.".format(total_mem))
    # exit(0)
    print(face_list)
    # table, edge_dict = allocate_table(face_list, start_edge, start_boundary_list, cont_sections)
    print("Finished setup: " + str(time.time()))
    count = count_non_int_paths(face_list, start_edge, exit_edge, start_boundary_list, cont_sections)
    print("Counted " + str(count) + " non-self-intersecting paths")
    print("Finish time: " + str(time.time()))
    return count


def test():
    from non_int_graph import setup_grid, number_faces
    correct_vals = [0, 0, 0, 2, 12, 184, 8512, 1262816, 575780564, 789360053252, 3266598486981642]
    global size
    for size in [4, 5, 6, 7]:
        grid = setup_grid(size, size)
        graph = nx.PlanarEmbedding()
        graph.set_data({v.name: [n.name for n in reversed(v.neighbors)] for v in grid.vertices})
        plt.subplot(111)
        nx.draw(graph)
        plt.show()
        face_dict = number_faces(graph, (size, size))
        face_list = [face_dict[k] for k in sorted(face_dict.keys())]
        exit_edge = (str(size - 1) + '.' + str(size - 2), str(size - 1) + '.' + str(size - 1))
        start_edge = ('0.0', '0.1')
        outer_face = max([graph.traverse_face(*exit_edge), graph.traverse_face(exit_edge[1], exit_edge[0])],
                         key=lambda x: len(x))
        start_boundary_list = []
        start_boundary_labels = []
        for i in range(len(outer_face)):
            edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
            if edge == exit_edge:
                continue
            start_boundary_list.append(edge)
            start_boundary_labels.append(1 if edge == ('0.0', '0.1') else 0)

        # start = time.time()
        positions = {x: [int(y) for y in x.split('.')] for x in graph.nodes}

        cont_sections = []
        face_list = []
        cur_edge_index = start_boundary_list.index(start_edge)
        cur_edge_index = start_boundary_list.index(
            (start_edge[1], start_edge[0])) if cur_edge_index == -1 else cur_edge_index
        cur_boundary = copy.deepcopy(start_boundary_list)

        # # Perform additional check to connect cont_sections we might have missed
        # # This can happen if a new face connects to multiple cont_sections, and we just add it to the first one
        # for i in range(len(cont_sections[-1])):
        #     # prev_ver[i]'s last vertex == prev_ver[i+1]'s first
        #     if cont_sections[-1][i][-1][1] == cont_sections[-1][(i + 1) % len(cont_sections[-1])][0][0]:
        #         # Connect!
        #         cont_sections[-1][i] += cont_sections[-1][(i + 1) % len(cont_sections[-1])]
        #         cont_sections[-1].pop((i + 1) % len(cont_sections[-1]))
        #         break
        # # cur_boundary = cur_boundary[:index] + new_loc + cur_boundary[index:]
        # face = ensure_ccw(face, positions)
        # face_list.append(face)
        print(face_list)
        print(cont_sections)
        var = count_non_int_paths_unrestr(face_list, start_edge, start_boundary_list, cont_sections)
        # print("Count for size {0} is {1}".format(size, count_non_int_paths(start_boundary, ('0.0',), dims, 0)))
        print(str(size) + ": " + str(var))
        # print("Duration: " + str(time.time() - start))
        assert var == correct_vals[size] if size < len(correct_vals) else ""


# https://gis.stackexchange.com/questions/281652/finding-all-neighbors-using-geopandas
def adjacency_from_shp(shapefile):
    # open file
    gdf = gpd.read_file(shapefile)
    gdf = gdf.to_crs(crs=3857)  # Distance calculation is done in meters
    g_data = collections.defaultdict(list)
    for index, precinct in gdf.iterrows():
        # get 'not disjoint' countries
        # would be queen adjacency
        neighbors = gdf[gdf.geometry.touches(precinct.geometry)]
        neighbors = neighbors[neighbors.geometry.intersection(precinct.geometry).length > .00001].index.tolist()
        # remove own name of the country from the list
        neighbors = [name for name in neighbors if index != name]
        g_data[index] = neighbors
    return g_data


# Makes an input graph planar by finding all intersection points of edges, and deleting one of the intersecting edges
# Uses an implementation of Bentley-Ottmann from https://github.com/ideasman42/isect_segments-bentley_ottmann
def find_intersecting_edges(g, positions):
    points = []
    for e in g.edges:
        if (tuple(positions[e[1]]), tuple(positions[e[0]])) not in points:
            points.append((tuple(positions[e[0]]), tuple(positions[e[1]])))
    inters = poly_point_isect.isect_segments_include_segments(points)
    inter_verts = []
    rev_positions = {}
    for vert, pos in positions.items():
        rev_positions[tuple(pos)] = vert
    for inter in inters:
        to_add1 = (rev_positions[inter[1][0][0]], rev_positions[inter[1][0][1]])
        to_add2 = (rev_positions[inter[1][1][0]], rev_positions[inter[1][1][1]])
        if to_add1 in inter_verts or to_add2 in inter_verts:
            continue
        inter_verts.append(to_add1)  # arbitrary
    return inter_verts


# 319441873429731761612648
# 58968945874956169986170552320
if __name__ == '__main__':
    # m_n = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467, 2356779, 6536382,
    #        18199284, 50852019, 142547559, 400763223, 1129760415, 3192727797, 9043402501, 25669818476, 73007772802,
    #        208023278209, 593742784829]
    # n = 29
    # d = depth_bound
    # for d in range(math.ceil(n/2)):
    #     print("Number of paths w/ depth <= {0}: {1}".format(d, m_n[n] - sum([1/(d2+1) * (math.comb(2*d2, d2) * math.comb(n, 2*d2)) for d2 in range(d, math.floor(n/2)+1)])))
    # for n in range(10, 30):
    #     print("Number of paths of length {0} w/ depth <= {1}: {2}".format(n, d, m_n[n] - sum([1/(d2+1) * (math.comb(2*d2, d2) * math.comb(n, 2*d2)) for d2 in range(d, math.floor(n/2)+1)])))
    # for d in range(math.ceil(n/2)):
    #     print("Number of paths w/ depth = {0}: {1}".format(d, 1/(d+1) * (math.comb(2*d, d) * math.comb(n, 2*d))))
    # face_order = [[46,48,130], [40,117,46], [46,130,48,128,40], [128,41,40], [132,128,48], [48,127,132], [127,128,132], [41,128,45], [136,128,127], [126,128,136], [136,127,48], [122,124,136], [136,129,126], [136,124,126,129], [126,45,128], [126,44,45], [124,44,126], [122,43,139,124], [139,43,141,124], [141,43,124], [43,120,124], [120,44,124], [120,42,44], [44,42,45], [42,41,45], [122,131,43], [43,131,120], [120,131,42], [42,131,121], [42,121,131,41], [122,136,48], [122,48,134], [164,122,134,48], [164,131,122], [164,97,131], [164,98,97], [164,87,98], [87,86,98], [87,31,86], [87,96,31], [164,96,87], [164,95,96], [96,95,31], [164,102,95], [164,103,95,102], [164,89,95,103], [164,161,89], [89,94,95], [89,161,94], [164,90,161], [166,91,90], [90,168,161], [90,169,161,168], [90,91,169], [169,91,161], [161,91,93,94], [95,94,93], [95,93,30,31], [93,29,30], [30,29,31], [29,88,86,31], [88,85,86], [151,88,29], [78,29,155], [73,78,155], [156,29,78], [73,156,78], [79,29,156], [73,79,156], [157,29,79], [73,157,79], [80,29,157], [73,80,157], [158,29,80], [73,158,80], [70,29,158], [73,70,158], [71,70,73], [69,70,71], [159,70,69], [159,29,70], [68,29,159], [68,159,69], [153,29,68,76], [153,76,68], [67,153,68], [67,68,69], [67,29,153], [67,152,160,151,29], [67,77,160,152], [67,160,77], [67,151,160], [67,58,151], [58,57,151], [61,67,69], [61,69,62], [60,67,61], [60,58,67], [56,61,62], [56,59,60,61], [56,58,59], [59,58,60], [56,57,58], [56,53,57], [55,53,56], [54,53,55], [55,56,62], [55,62,64], [64,62,63], [64,63,65], [63,62,65], [62,144,65], [62,69,144], [65,144,66], [144,69,66], [66,69,71], [72,66,71], [74,73,75], [74,71,73]]
    # Unit test for find_motzkin_paths
    # my_dict = {}
    # find_motzkin_paths(0, '', 5, my_dict, 0, True)
    # print(my_dict)
    # exit(0)
    face_order = [[46, 48, 130], [46, 130, 48, 128, 40], [46, 116, 47, 50], [46, 47, 116], [46, 117, 47], [46, 40, 117],
                  [117, 40, 47], [128, 41, 40], [41, 131, 47, 40], [131, 52, 50, 47], [50, 52, 51], [51, 52, 106],
                  [52, 13, 106], [11, 13, 52, 22], [11, 22, 12], [22, 52, 23], [23, 52, 21], [21, 52, 131, 97],
                  [22, 23, 21], [12, 22, 35, 104], [22, 21, 35], [35, 21, 97], [132, 128, 48], [48, 127, 132],
                  [127, 128, 132], [41, 128, 45], [136, 128, 127], [126, 128, 136], [136, 127, 48], [122, 124, 136],
                  [136, 129, 126], [136, 124, 126, 129], [126, 45, 128], [126, 44, 45], [124, 44, 126],
                  [122, 43, 139, 124], [139, 43, 141, 124], [141, 43, 124], [43, 120, 124], [120, 44, 124],
                  [120, 42, 44], [44, 42, 45], [42, 41, 45], [42, 121, 131, 41], [42, 131, 121], [120, 131, 42],
                  [43, 131, 120], [122, 131, 43], [122, 136, 48], [122, 48, 134], [164, 122, 134, 48], [164, 131, 122],
                  [164, 97, 131], [164, 98, 97], [98, 86, 84, 97], [84, 35, 97], [83, 35, 84], [85, 83, 84],
                  [86, 85, 84], [151, 83, 85, 88], [86, 88, 85], [82, 35, 83], [81, 35, 82], [82, 83, 53], [81, 82, 53],
                  [54, 81, 53], [55, 54, 53], [53, 83, 57], [57, 83, 151], [151, 88, 29], [29, 88, 86, 31],
                  [87, 31, 86], [87, 96, 31], [87, 86, 98], [164, 87, 98], [164, 96, 87], [164, 95, 96], [96, 95, 31],
                  [95, 93, 30, 31], [30, 29, 31], [93, 29, 30], [55, 53, 56], [56, 53, 57], [56, 57, 58], [58, 57, 151],
                  [56, 58, 59], [56, 59, 60, 61], [59, 58, 60], [60, 58, 67], [60, 67, 61], [67, 58, 151],
                  [67, 151, 160], [67, 160, 77], [67, 77, 160, 152], [67, 152, 160, 151, 29], [67, 29, 153],
                  [67, 153, 68], [153, 76, 68], [153, 29, 68, 76], [67, 68, 69], [68, 159, 69], [68, 29, 159],
                  [159, 29, 70], [159, 70, 69], [61, 67, 69], [61, 69, 62], [56, 61, 62], [55, 56, 62], [55, 62, 64],
                  [64, 62, 63], [64, 63, 65], [63, 62, 65], [65, 62, 144], [62, 69, 144], [65, 144, 66], [144, 69, 66],
                  [66, 69, 71], [72, 66, 71], [69, 70, 71], [164, 102, 95], [164, 103, 95, 102], [164, 89, 95, 103],
                  [89, 94, 95], [95, 94, 93], [164, 161, 89], [89, 161, 94], [161, 91, 93, 94], [91, 92, 93],
                  [92, 29, 93], [164, 90, 161], [90, 168, 161], [90, 169, 161, 168], [169, 91, 161], [90, 91, 169],
                  [33, 90, 164], [33, 165, 90], [33, 166, 90, 165], [166, 91, 90], [33, 91, 166], [33, 92, 91],
                  [33, 114, 92], [177, 29, 92, 115], [177, 115, 92], [114, 177, 92], [33, 34, 32, 114], [34, 112, 32],
                  [101, 147, 177, 114], [101, 177, 147], [170, 101, 114, 32], [37, 170, 32], [112, 37, 32],
                  [112, 36, 37], [38, 170, 37], [36, 38, 37], [39, 38, 36], [112, 39, 36], [110, 39, 112],
                  [110, 38, 39], [110, 170, 38], [146, 101, 170], [146, 178, 101], [146, 179, 101, 178],
                  [146, 180, 101, 179], [146, 154, 176, 177, 101, 180], [146, 176, 154], [146, 173, 177, 176],
                  [146, 177, 173], [155, 29, 177], [78, 29, 155], [73, 78, 155], [156, 29, 78], [73, 156, 78],
                  [79, 29, 156], [73, 79, 156], [157, 29, 79], [73, 157, 79], [80, 29, 157], [73, 80, 157],
                  [158, 29, 80], [73, 158, 80], [70, 29, 158], [73, 70, 158], [71, 70, 73], [146, 73, 155, 177],
                  [146, 75, 73], [146, 149, 75], [146, 174, 149], [174, 182, 149], [146, 182, 174], [146, 171, 182],
                  [171, 172, 182], [182, 172, 149], [172, 74, 149], [149, 74, 75], [75, 74, 73], [74, 71, 73]]
    # adjacency_from_shp("data/exp2627wards.shp")
    random.seed(123456)
    np.random.seed(123456)
    # gdf = gpd.read_file("/home/dani/PycharmProjects/GerryMand/data/wi_cbgs/wi_pl2020_bg.shp", driver='ESRI shapefile', encoding='UTF-8')
    # ngdf = gdf.to_numpy()
    #
    # def mode(a):
    #     u, c = np.unique(a, return_counts=True)
    #     return u[c.argmax()]
    #
    # def cust_agg(series):
    #     return "sum" if series.dtype == "int64" else "mode"
    #
    # gdf = gdf.dissolve(by="TRACT", aggfunc="cust_agg")
    # enumerate_paths_with_order("data/exp2627wards.shp", face_order, draw=False, recalculate=True)
    enumerate_paths_with_order("data/wi_cbgs/wi_pl2020_bg.shp", face_order, draw=False, recalculate=True)
    # enumerate_paths("data/exp2627neighb.dbf", "data/exp2627wards.shp")
    # test()
    # out_dict = {}
    # find_motzkin_paths(0, '', 6, out_dict, 2)
    # print(out_dict)
    exit()
