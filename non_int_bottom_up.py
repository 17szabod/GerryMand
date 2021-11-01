# Core idea: Enumerate Motzkin paths into preallocated data structure
# Each contiguous section of the frontier has its own motzkin paths
# The 1 could be anywhere in the start_edge's contiguous section
# Maintain a dictionary of edge relations
# Motzkin paths should be enumerated by putting in either a 0 or a 3 or 2 if 3 is before
# List of dicts?
# Only main algorithm is bottom up, rest can be top down

import collections
import copy
import gc
import itertools
import math
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

debug = False
depth_bound = 10


# https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partition(number, p_count):
    answer = set()
    if p_count == 1:
        answer.add((number, ))
        return answer
    for x in range(0, number + 1):
        for y in partition(number - x, p_count - 1):
            answer.add((x, ) + y)
    return answer


def count_non_int_paths_w_table(table, edge_dict):
    table[-1][str(len(table)) + '.1'] = 1
    for i in range(1, len(table)):  # Go through bottom up (always subtract i)
        for path in table[len(table) - i - 1]:
            table[len(table) - i - 1][path] = sum(
                [table[len(table) - i][edge_dict[x][edge_dict[x].index('.') + 1:]] for x in
                 edge_dict[str(len(table) - i) + '.' + path]])
    return table


# @profile
def count_non_int_paths(face_list, start_edge, outer_boundary, cont_sections):
    # Loop through reversed cont_sections - keep track of cur_sect and prev_sect
    # Generate and loop through each motzkin path of cur_sect and find connected path in prev_sect
    # Add the values of each of cur_sect path's neighbors from prev_sect to it's value
    # Store these in prev_dict and cur_dict
    # prev_dict = cur_dict - manually gc.collect() after?
    one = '1' if start_edge == (46,48) else '0'
    prev_dict = {one: 1}  # Do we need to store which step it is in these dicts? No!
    prev_sect = cont_sections[-1]
    flat_outer = [x[0] for x in outer_boundary] + [outer_boundary[-1][1]]
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    start_edge_ind = outer_boundary.index(start_edge)
    num_samples = 1000
    sample_paths = [[one] for x in range(num_samples)]  # The sampled path to return
    # sample_path = ['1']
    sample_tree = [{one: [1, []]}]
    for i in range(2, len(cont_sections)+1):  # Range is off because negative values are offset
        cur_sect = cont_sections[-i]
        cur_dict = collections.defaultdict()
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
                        for start_depth in range(tup[section_ind]+1):  # will repeat, but repeats are ok
                            first_dict = {}
                            second_dict = {}
                            find_motzkin_paths(0, '', len(section) - one_ind - 1, first_dict, depth_bound - start_depth)
                            find_motzkin_paths(0, '', one_ind, second_dict, depth_bound - (tup[section_ind] - start_depth))
                            my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                    temp_dict = {s1 + s2: 0 for s1 in temp_dict.keys() for s2 in my_dict.keys()}
                else:
                    my_dict = {}
                    find_motzkin_paths(0, '', len(section), my_dict, depth_bound - tup[section_ind])
                    temp_dict = {s1 + s2: 0 for s1 in temp_dict.keys() for s2 in my_dict.keys()}
            cur_dict.update(temp_dict)
        face = face_list[-i+1]
        label_inds = []  # Inds in flattened_sections
        labeled_edges = []  # List of edges that have labels to make searching later easier
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
        mapping = [trimmed_prev_flattened_sections.index(flattened_sections[j]) for j in range(len(flattened_sections)) if flattened_sections[j] not in labeled_edges]
        mapping = [mapping.index(x) for x in range(len(mapping))]  # need to invert mapping, might be faster to do it above but speed doesnt matter in this part
        path_map = collections.defaultdict(list)  # A mapping of paths to their "neighbors" to help with sampling
        print("Working on section {0} with length {1}".format(len(cont_sections)-i, len(flattened_sections))) if debug else ""
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
                path_map[path1].append(path) if path1 in prev_dict else ""
                for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                    string_to_add = '0' * ind1 + '3' + '0' * (ind2 - ind1 - 1) + '2' + '0' * (len(new_loc) - ind2 - 1)
                    path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                    # path1 = next_path[:index] + string_to_add + next_path[index:]
                    cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
                    path_map[path1].append(path) if path1 in prev_dict else ""
                    # if path1 not in prev_dict:
                    #     print(path1)
            elif len(labels) == 1:
                for ind1 in range(len(new_loc)):
                    string_to_add = '0' * ind1 + labels[0] + '0' * (len(new_loc) - 1 - ind1)
                    path1 = insert_at_indices(next_path, string_to_add, inds_to_add)
                    # path1 = next_path[:index] + string_to_add + next_path[index:]
                    cur_dict[path] += prev_dict[path1] if path1 in prev_dict else 0
                    path_map[path1].append(path) if path1 in prev_dict else ""
                    # if path1 not in prev_dict:
                    #     print(path1)
            elif labels in [('1', '2'), ('2', '1'), ('1', '3'), ('3', '1'), ('2', '2'), ('3', '3')]:
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
                    for x in range(index-1, -1, -1):
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
                path_map[path1].append(path) if path1 in prev_dict else ""
                # if path1 not in prev_dict:
                #     print(path1)
            elif labels == ('3', '2') or labels == ('2', '3'):
                pass  # ?
                # print("Closed a loop!")
                # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
                # return 0
                # We just closed a loop! Currently allowed
                # raise Exception("Theoretically impossible case occurred, we closed a loop.")
            else:
                raise Exception("Invalid labels on step location")
        # Sample by stepping backwards - this is only possible once we've already created both prev and cur dicts
        # This might work, but stepping backwards is nontrivial - for now, slow down above
        # Sol: keep track of PARTIAL full trees, and sample once space is too large
        # offset = 0
        # for i in range(len(sample_paths)):
        #     sample_path = sample_paths[i - offset]
        #     count_arr = [cur_dict[x] for x in path_map[sample_path[-1]]]
        #     if len(count_arr) == 0:
        #         sample_paths.pop(i - offset)
        #         offset +=1
        #         continue
        #     choice = np.random.uniform(0, sum(count_arr))
        #     sample_ind = np.arange(len(count_arr))[np.asanyarray([sum(count_arr[:x+1]) >= choice for x in range(len(count_arr))])][0]
        #     sample_paths[i-offset].append(path_map[sample_path[-1]][sample_ind])
        # Build another layer of sample_tree
        new_layer = collections.defaultdict(list)
        for p in sample_tree[-1]:
            sample_tree[-1][p][1] += path_map[p]
            for m in path_map[p]:
                new_layer[m] = [cur_dict[m], []]
        sample_tree.append(new_layer)
        # clean empty paths
        prev_rem = []
        for i2 in range(len(sample_tree)-1):
            to_rem = []
            cur_layer = sample_tree[-i2-2]
            for p in cur_layer:
                offset = 0
                for j in range(len(cur_layer[p][1])):
                    if cur_layer[p][1][j-offset] in prev_rem:
                        cur_layer[p][1].pop(j-offset)
                        offset += 1
                if len(cur_layer[p][1]) == 0:
                    to_rem.append(p)
            for p in to_rem:
                cur_layer.pop(p)
            prev_rem = copy.deepcopy(to_rem)
        # Occasionally trim!
        if len(sample_tree) >= 25 or i==len(cont_sections):
            for k in range(len(sample_tree)-1):
                layer = sample_tree[k]
                offset = 0
                for i2 in range(len(sample_paths)):
                    sample_path = sample_paths[i2 - offset]
                    # Need to find keep track of counts too
                    if sample_path[-1] not in layer:  # Try to take a step back and salvage it?
                        if sample_path[-2] in sample_tree[k-1] and k > 0:
                            count_arr = [sample_tree[k][x][0] for x in sample_tree[k-1][sample_path[-2]][1]]
                            if len(count_arr) == 0:
                                sample_paths.pop(i2 - offset)
                                offset += 1
                                continue
                            choice = np.random.uniform(0, sum(count_arr))
                            sample_ind = np.arange(len(count_arr))[
                                np.asanyarray([sum(count_arr[:x + 1]) >= choice for x in range(len(count_arr))])][0]
                            sample_paths[i2 - offset].pop(-1)
                            sample_paths[i2 - offset].append(sample_tree[k-1][sample_path[-2]][1][sample_ind])
                        else:
                            sample_paths.pop(i2 - offset)
                            offset += 1
                            continue
                    count_arr = [sample_tree[k+1][x][0] for x in layer[sample_path[-1]][1]]
                    if len(count_arr) == 0:
                        sample_paths.pop(i2 - offset)
                        offset += 1
                        continue
                    choice = np.random.uniform(0, sum(count_arr))
                    sample_ind = np.arange(len(count_arr))[np.asanyarray([sum(count_arr[:x+1]) >= choice for x in range(len(count_arr))])][0]
                    sample_paths[i2-offset].append(layer[sample_path[-1]][1][sample_ind])
            new_layer = collections.defaultdict(list)
            for i2 in range(len(sample_paths)):
                new_layer[sample_paths[i2][-1]] = [cur_dict[sample_paths[i2][-1]], []]
            sample_tree = [new_layer]
        prev_dict = cur_dict
        prev_sect = cur_sect
    # for path in sample_paths:
    #     path += ['1']
    return list(prev_dict.values())[0], sample_paths  # There should only be one value at the end


# https://doi.org/10.1016/j.tcs.2020.12.013
def find_motzkin_paths(h, w, n, m_dict, depth):
    j = len(w)
    if depth > depth_bound:
        return
    if h > n - j:
        return
    if j > n:
        m_dict[w] = 0
        return
    if h == n - j:
        m_dict[w + h * '2'] = 0
        return
    if h > 0:
        find_motzkin_paths(h - 1, w + '2', n, m_dict, depth)
    find_motzkin_paths(h, w + '0', n, m_dict, depth)
    find_motzkin_paths(h + 1, w + '3', n, m_dict, depth+1)
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
    for i in range(2, len(cont_sections)+1):  # Range is off because negative values are offset
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
        face = face_list[-i+1]
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
        mapping = [trimmed_prev_flattened_sections.index(flattened_sections[j]) for j in range(len(flattened_sections)) if flattened_sections[j] not in labeled_edges]
        mapping = [mapping.index(x) for x in range(len(mapping))]  # need to invert mapping, might be faster to do it above but speed doesnt matter in this part
        print("Working on section {0} with length {1}".format(len(cont_sections)-i, '.'.join([str(len(sect)) for sect in cur_sect])))
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


def allocate_table(face_list, start_edge, outer_boundary, cont_sections):
    # Put the one in first
    # How should I keep track of contiguous sections? - build up
    # How to find edge relations? - top down and reverse
    # How should I enumerate all Motzkin paths? - Use recursive method to create strings for each section and then
    # take the cartesian product
    big_table = []
    flat_outer = [x[0] for x in outer_boundary] + [outer_boundary[-1][1]]
    outer_boundary = [tuple(sorted(x)) for x in outer_boundary]
    start_edge_ind = outer_boundary.index(start_edge)
    for sections in cont_sections:
        path_dict = {}
        # gc.collect()
        for section in sections:
            # find section that connects to start_edge
            start_ind = flat_outer.index(section[0][0]) if section[0][0] in flat_outer else flat_outer.index(
                section[0][1])
            end_ind = flat_outer.index(section[-1][0]) if section[-1][0] in flat_outer else flat_outer.index(
                section[-1][1])
            if start_ind <= start_edge_ind <= end_ind or (len(sections) == 1 and len(big_table) > len(outer_boundary)):
                my_dict = {}
                for i in range(len(section)):
                    first_dict = {}
                    second_dict = {}
                    find_motzkin_paths(0, '', len(section) - i, first_dict, depth_bound)
                    find_motzkin_paths(0, '', i, second_dict, depth_bound)
                    my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                path_dict = {s1 + s2: 0 for s1 in path_dict.keys() for s2 in my_dict.keys()}
            else:
                my_dict = {}
                find_motzkin_paths(0, '', len(section), my_dict, depth_bound)
                path_dict = {s1 + s2: 0 for s1 in path_dict.keys() for s2 in my_dict.keys()}
        big_table.append(path_dict)
    # Generate edge relations
    edge_dict = collections.defaultdict()
    for i in range(len(cont_sections) - 1):
        face = face_list[i]
        label_inds = []
        new_loc = []
        # Find index of step
        flattened_sections = [x for j in range(len(cont_sections[i])) for x in cont_sections[i][j]]
        index = sum([len(cont_sections[i][j]) for j in range(len(cont_sections[i]))])
        for j in range(len(face)):
            edge = (face[((j + 1) % len(face))], face[j])  # We know it'll be reversed!
            if edge in outer_boundary:
                pass
            elif edge in flattened_sections:
                cur_index = flattened_sections.index(edge)
                label_inds.append(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append((edge[1], edge[0]))
        for path in big_table[i].keys():
            # Find step type (labels)
            labels = tuple([int(path[x]) for x in label_inds if path[x] != '0'])
            next_path = ''.join([path[x] for x in range(len(path)) if x not in label_inds])
            # Add edge to all possible consequences of labels
            if len(labels) == 0:
                path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                edge_dict[str(i + 1) + '.' + path1].append(str(i) + '.' + path)
                for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # this preserves order!
                    string_to_add = '0' * (ind1 - 1) + '3' + '0' * (ind2 - ind1 - 1) + '2' + '0' * (len(new_loc) - ind2)
                    path1 = next_path[:index] + string_to_add + next_path[index:]
                    edge_dict[str(i + 1) + '.' + path1].append(str(i) + '.' + path)
            elif len(labels) == 1:
                for ind1 in range(len(new_loc)):
                    string_to_add = '0' * (ind1 - 1) + str(labels[0]) + '0' * (len(new_loc) - ind1)
                    path1 = next_path[:index] + string_to_add + next_path[index:]
                    edge_dict[str(i + 1) + '.' + path1].append(str(i) + '.' + path)
            elif labels in [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (3, 3), (2, 3)]:
                path1 = next_path[:index] + '0' * len(new_loc) + next_path[index:]
                count = 0
                if labels == (2, 3):  # possible, just combine
                    pass
                # Need to find partner and change label:
                elif 3 in labels:  # 2 will be below it
                    for x in range(index, len(flattened_sections)):
                        if path[x] == 3:
                            count += 1
                        if path[x] == 2:
                            if count != 0:
                                count -= 1
                            else:
                                path1 = path1[:x] + '1' if labels != (3, 3) else '3' + path1[x + 1:]
                else:
                    for x in range(index, 0, -1):
                        if path[x] == 2:
                            count += 1
                        if path[x] == 3:
                            if count != 0:
                                count -= 1
                            else:
                                path1 = path1[:x] + '1' if labels != (2, 2) else '2' + path1[x + 1:]
                if count != 0:
                    raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")
                edge_dict[str(i + 1) + '.' + path1].append(str(i) + '.' + path)
            elif labels == (3, 2):
                pass  # ?
                # print("Closed a loop!")
                # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
                # return 0
                # We just closed a loop! Currently allowed
                # raise Exception("Theoretically impossible case occurred, we closed a loop.")
            else:
                raise Exception("Invalid labels on step location")
    return big_table, edge_dict


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
    centers = loc_df.set_geometry('centroid_column')
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
        plt.figure(figsize=(25, 25))
        nx.draw(g, pos=positions, node_size=60, with_labels=True, font_size=12, font_color='red', linewidths=0, width=.2)
        # G2 = h.subgraph([106,0,3,4,5,6,7,8,9,14,107,13,51,52])
        # nx.draw(G2, pos=positions, with_labels=True)
        # centers.plot()
        # loc_df.plot()
        plt.show()
        exit()
    g.check_structure()
    print(order_faces(g, positions))


def enumerate_paths_with_order(adj_file, shapefile, face_order, draw=True):
    print("Start time: " + str(time.time()))
    df = gpd.read_file(adj_file)
    np_df = df.to_numpy()
    g_data = collections.defaultdict(list)
    for i in range(len(np_df)):
        g_data[np_df[i][0]].append(np_df[i][1]) if np_df[i][2] > 0.00001 else ""
    loc_df = gpd.read_file(shapefile, driver='ESRI shapefile', encoding='UTF-8')
    loc_df['centroid_column'] = loc_df.centroid
    centers = loc_df.set_geometry('centroid_column')
    # centers.set_index('OBJECTID', inplace=True)
    # print(centers)
    h = nx.DiGraph(incoming_graph_data=g_data)
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
    g.set_data(oriented_g_data)
    success, counterexample = nx.check_planarity(g, counterexample=True)
    if not success:
        nx.draw(counterexample, pos=positions, with_labels=True)
        plt.show()
        print("Error: Adjacency graph is not planar, exiting...")
        exit(0)
    g.check_structure()

    exit_edge = (71, 74)
    start_edge = (46, 48)
    sindex = 0
    eindex = 0
    outer_face = max([g.traverse_face(*exit_edge), g.traverse_face(exit_edge[1], exit_edge[0])],
                     key=lambda x: len(x))
    sample_paths = []
    for i in range(len(outer_face)):
        # edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
        edge = (outer_face[i], outer_face[(i + 1) % len(outer_face)])
        if edge == start_edge:
            sindex = i
        if edge == exit_edge:
            eindex = i
    cts = []
    diversities = []
    efficiencies = []
    for i in range(int(len(outer_face)/2)):
        start_edge = (outer_face[(sindex+i) % len(outer_face)], outer_face[(sindex + i + 1) % len(outer_face)])
        exit_edge = (outer_face[(eindex+i) % len(outer_face)], outer_face[(eindex + i + 1) % len(outer_face)])
        print("Sampling with start edge {0} and exit edge {1}".format(start_edge, exit_edge))
        cont_sections, count, sample_paths = count_and_sample(draw, face_order, g, positions, exit_edge, start_edge)
        for path in sample_paths:
            ct, g1, g2 = eval_path(path, cont_sections, copy.deepcopy(g), positions, start_edge, exit_edge, draw2=False)
            sum1 = 0
            sum2 = 0
            for v in g1:
                sum1 += loc_df.loc[v]['PERSONS']
            for v in g2:
                sum2 += loc_df.loc[v]['PERSONS']
            if math.fabs(sum1 - sum2) < 1000:
                # sum1 = 0
                # sum2 = 0
                efficiencies.append(calculate_eff_gap(g1, g2, loc_df, sum1, sum2))
                # print("{0}, {1}".format(sum1, sum2))
            cts.append(ct)
        print("Found {0} possible partitions".format(len(efficiencies)))
    print("Path length distribution: Mean {0} with variance {1}, shortest path {2}".format(np.mean(cts), np.std(cts), np.min(cts)))
    # print("Diversity distribution: Sampled {2} with mean {0} with variance {1}".format(np.mean(diversities), np.std(diversities), len(diversities)))
    print("Efficiency gap distribution: Sampled {2} with mean {0} with variance {1}".format(np.mean(efficiencies), np.std(efficiencies), len(efficiencies)))
    sum1 = 0
    sum2 = 0
    g1 = []
    g2 = []
    for v in g.nodes:
        if loc_df.loc[v]['DISTRICT'] == '27':
            # sum1 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
            sum1 += loc_df.loc[v]['PERSONS']
            g1.append(v)
        else:
            # sum2 += loc_df.loc[v]['PERSONS'] - loc_df.loc[v]['WHITE']
            sum2 += loc_df.loc[v]['PERSONS']
            g2.append(v)
    # print("True diversity was: " + str(math.fabs(sum2 - sum1)))
    print("True efficiency gap was: " + str(calculate_eff_gap(g1, g2, loc_df, sum1, sum2)))
    print("Finish time: " + str(time.time()))
    return count


def count_and_sample(draw, face_order, g, positions, exit_edge, start_edge):
    # exit_edge = (71, 74)
    # start_edge = (46, 48)
    # Order the faces according to face_order
    outer_face = max([g.traverse_face(*exit_edge), g.traverse_face(exit_edge[1], exit_edge[0])],
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
    for edge in g.edges:
        sorted_edge = sorted(edge, key=lambda x: positions[x][1])  # unnecessary to do this I think
        face = g.traverse_face(sorted_edge[1], sorted_edge[0])  # Traverse clockwise
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
                    or (exit_edge[0] not in inside_slice and exit_edge[1] not in inside_slice):
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
    h2 = g.subgraph(list(verts_left)).copy()
    if draw:
        # ax = plt.subplot(121)
        # plt.sca(ax)
        plt.figure(figsize=(25, 25))
        # nx.draw(g, pos=positions, node_size=60, with_labels=True, font_size=12, font_color='red', linewidths=0, width=.2)
        nx.draw(h2, pos=positions, node_size=60, with_labels=True, font_size=12, font_color='red', linewidths=0,
                width=.2)
        # nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=6, font_color='red')
        # G2 = h.subgraph([106,0,3,4,5,6,7,8,9,14,107,13,51,52])
        # nx.draw(G2, pos=positions, with_labels=True)
        # centers.plot()
        # loc_df.plot()
        plt.show()
        # exit()
    # Test that face_ordr facesactually exist
    for face in face_order:
        oface = ensure_cw(face, positions)
        sedges = [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
        oedges = [(oface[i], oface[(i + 1) % len(oface)]) for i in range(len(oface))]
        for f in sedges:
            if f not in oedges:
                print("This face is oriented incorrectly: " + str(face) + ", not like " + str(oface))
                break
    for face in face_order:
        # f_set = [[face[(i+k)%len(face)] for i in range(len(face))] for k in range(len(face))]
        f_set = set(face)
        d_sets = [set(f) for f in face_dict.values()]
        if f_set not in d_sets:
            print("This face was in face_order but not in dict: " + str(face))
    for face in face_dict.values():
        f_set = set(face)
        d_sets = [set(f) for f in face_order]
        if f_set not in d_sets:
            print("This face was in dict but not in face_order: " + str(face))
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
    # Ensure that face_list results in a continuous boundary
    # Iterate through face_list keeping track of contiguous boundary sets
    # Store the contiguous sections as a list (each frontier) of lists (each connected component) of lists (frontiers)
    cont_sections = []
    face_list = []
    cur_boundary = copy.deepcopy(start_boundary_list)
    for face in face_order:
        print(face) if debug else ""
        # if face == [43,131,120]:
        #     print('h')
        print(cur_boundary) if debug else ""
        face = list(reversed(face))
        # Set up vertices in boundary
        boundary_verts = [x[0] for x in cur_boundary] + [cur_boundary[-1][1]]
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
            edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
            if edge in cur_boundary:
                cur_index = cur_boundary.index(edge)
                cur_boundary.pop(cur_index)
                if cur_index < index:
                    index = cur_index
            else:
                new_loc.append((edge[1], edge[0]))
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
                    # have to watch out for when we are cutting on the boundary!
                    # if new_loc[0][0] in
                    cont_sections += [prev_ver]
                    cont_sections[-1][i] = new_loc + prev_ver[i][section_vert.index(new_loc[-1][1]):]
                    new_section = False
                    break
            if new_section:
                prev_ver.append(new_loc)
                cont_sections.append(prev_ver)
        print([[edge[0] for edge in section] + [section[-1][1]] for section in cont_sections[-1]])
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
    # exit(0)
    print(face_list)
    # table, edge_dict = allocate_table(face_list, start_edge, start_boundary_list, cont_sections)
    print("Finished setup: " + str(time.time()))
    # count = count_non_int_paths_w_table(table, edge_dict)
    count, sample_paths = count_non_int_paths(face_list, start_edge, start_boundary_list, cont_sections)
    print("Counted " + str(count) + " non-self-intersecting paths")
    return cont_sections, count, sample_paths


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


def eval_path(path, cont_sections, g, positions, start_edge, exit_edge, draw=False, draw2=False):
    edges = {start_edge, exit_edge, (start_edge[1], start_edge[0]), (exit_edge[1], exit_edge[0])}
    for i in range(len(path)):
        assignment = path[-i-1]
        sect = cont_sections[i]
        flat_sect = [x for j in range(len(sect)) for x in sect[j]]
        edges = edges.union(set([flat_sect[i] for i in range(len(assignment)) if assignment[i] != '0']))
        edges = edges.union(set([(flat_sect[i][1], flat_sect[i][0]) for i in range(len(assignment)) if assignment[i] != '0']))
    if draw:
        edge_col = [('blue' if e in edges else 'black') for e in g.edges]
        nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=6, font_color='red', edge_color=edge_col)
        plt.show()
    g.remove_edges_from(edges)
    if draw2:
        plt.figure(figsize=(18, 18))
        nx.draw(g, pos=positions, node_size=60, with_labels=True, font_size=12, font_color='red', linewidths=0, width=.2)
        # nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=6, font_color='red')
        plt.show()
    comps = list(nx.connected_components(g))
    return len(edges), comps[0], comps[1]


def order_faces(graph, positions):
    # Construct boundaries
    exit_edge = (71, 74)
    start_edge = (46, 48)

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
            print(str(int(x/2))+', ', end='')
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
                cur_edge_index = (cur_edge_index + 3) % (len(cur_boundary) - 1) if pass_counter < len(outer_face)*20 else \
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
            sect_mem *= sum([1/(d2+1) * (math.comb(2*d2, d2) * math.comb(len(sect), 2*d2)) for d2 in range(depth_bound+1)])
        print("{0}: {1}: {2}".format(c, '.'.join([str(len(sect)) for sect in sections]), sect_mem))
        c += 1
        total_mem += sect_mem
    print("Will be using approximately {0} entries.".format(total_mem))
    # exit(0)
    print(face_list)
    # table, edge_dict = allocate_table(face_list, start_edge, start_boundary_list, cont_sections)
    print("Finished setup: " + str(time.time()))
    count = count_non_int_paths(face_list, start_edge, start_boundary_list, cont_sections)
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

        for i in range(size):
            for j in range(size-1):
        #         cur_edge = [str(i) + '.' + str(j), str(i) + '.' + str(j+1)]
        #         print(cur_edge)
        #         print(cur_boundary)
        #         face = graph.traverse_face(cur_edge[1], cur_edge[0])
        #         # face = ensure_ccw(face, positions)
        #         # boundary_verts = [x[0] for x in cur_boundary] + [cur_boundary[-1][1]]
        #         # Face is good! Add to boundary
        #         # stores the new vertices that will be added to the boundary
        #         new_loc = []
        #         # stores the index all face elements will be put to
        #         index = len(cur_boundary)
        #         for i in range(len(face)):
        #             edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
        #             named_edge = tuple(sorted(edge))
        #             if named_edge in cur_boundary:
        #                 cur_index = cur_boundary.index(named_edge)
        #                 cur_boundary.pop(cur_index)
        #                 if cur_index < index:
        #                     index = cur_index
        #             else:
        #                 new_loc.append(named_edge)
        #         # Adjust order of edges in new_loc to ensure beginning and end is good
        #         new_loc_copy = []
        #         for i in range(len(new_loc)):
        #             if i > 0:
        #                 new_loc_copy.append(new_loc[i] if new_loc[i][0] in new_loc[i-1] else (new_loc[i][1], new_loc[i][0]))
        #             elif i < len(new_loc) - 1:
        #                 new_loc_copy.append(
        #                     new_loc[i] if new_loc[i][1] in new_loc[i + 1] else (new_loc[i][1], new_loc[i][0]))
        #             else:
        #                 new_loc_copy.append(new_loc[i])
        #         new_loc = new_loc_copy
        #         # Make sure new_loc follows the order of boundary:
        #         swapped = len(cur_boundary) > 0 and len(new_loc) > 0 and \
        #                   (len(cur_boundary) > index > 0 and (
        #                           len(set(new_loc[0]).intersection(set(cur_boundary[index - 1]))) == 0 or len(
        #                       set(new_loc[-1]).intersection(set(cur_boundary[index])))) == 0)
        #         if swapped:  # need to find where to start the new locations
        #             start_ind = -1
        #             for i in range(len(new_loc)):
        #                 if len(set(new_loc[i]).intersection(set(cur_boundary[index - 1]))) > 0:
        #                     start_ind = i
        #                     if len(set(new_loc[i]).intersection(set(cur_boundary[index - 1]))) == 1 and len(
        #                             set(new_loc[(i - 1) % len(new_loc)]).intersection(set(cur_boundary[index]))) == 1:
        #                         print("Found good rotation: " + str(new_loc)) if debug else ""
        #                         break
        #             new_loc = [new_loc[(x + start_ind) % len(new_loc)] for x in range(len(new_loc))]
        #         # Find which parts of cont_sections new_loc matches with
        #         # Need to merge sections/ remove them
        #         if len(cont_sections) == 0:  # special first-time setup
        #             cont_sections.append([new_loc])
        #         elif len(new_loc) == 0:  # Another special case, just remove appropriate edges
        #             prev_ver = copy.deepcopy(cont_sections[-1])
        #             for j in range(len(prev_ver)):
        #                 for i in range(len(face)):
        #                     edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
        #                     if edge in prev_ver[j]:
        #                         prev_ver[j].remove(edge)
        #                         if len(prev_ver[j]) == 0:
        #                             prev_ver.pop(j)
        #             cont_sections.append(prev_ver)
        #         else:
        #             new_section = True
        #             prev_ver = copy.deepcopy(cont_sections[-1])
        #             cont_section_verts = [[edge[0] for edge in section] + [section[-1][1]] for section in cont_sections[-1]]
        #             # new_loc_verts = [edge[0] for edge in new_loc] + [new_loc[-1][1]]
        #             for i in range(len(cont_section_verts)):
        #                 section_vert = cont_section_verts[i]
        #                 truth_table = (new_loc[0][0] in section_vert, new_loc[-1][1] in section_vert)
        #                 if truth_table == (True, True):  # Just add to current cont_section
        #                     cont_sections += [prev_ver]
        #                     cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc \
        #                                            + prev_ver[i][section_vert.index(new_loc[-1][1]):]
        #                     new_section = False
        #                     break
        #                 elif truth_table == (False, False):
        #                     continue
        #                 elif truth_table == (True, False):  # Need to connect two sections to create a new one - or extend!
        #                     should_extend = True
        #                     for k in range(len(cont_section_verts)):
        #                         if new_loc[-1][1] in cont_section_verts[(i + k) % len(cont_section_verts)]:
        #                             prev_ver[i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
        #                                           prev_ver[(i + k) % len(prev_ver)][
        #                                           cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]
        #                             prev_ver.pop((i + k) % len(prev_ver))
        #                             # cont_sections += [prev_ver[:i]] if i + k != len(cont_section_verts) else [prev_ver[1:i]]
        #                             # cont_sections[-1] += [prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc + \
        #                             #                       prev_ver[(i + k) % len(prev_ver)][
        #                             #                       cont_section_verts[(i + k) % len(prev_ver)].index(new_loc[-1][1]):]]
        #                             # cont_sections[-1] += prev_ver[i + k + 1:]
        #                             cont_sections += [prev_ver]
        #                             should_extend = False
        #                     if should_extend:  # Extend!
        #                         cont_sections += [prev_ver]
        #                         cont_sections[-1][i] = prev_ver[i][:section_vert.index(new_loc[0][0])] + new_loc
        #                     new_section = False
        #                     break
        #                 elif truth_table == (False, True):  # Extend at the beginning- only if new_loc[0][0] nowhere else
        #                     should_continue = False
        #                     for k in range(len(cont_section_verts)):
        #                         if new_loc[0][0] in cont_section_verts[k]:
        #                             should_continue = True
        #                     if should_continue:  # Very awkward way to skip
        #                         continue
        #                     cont_sections += [prev_ver]
        #                     cont_sections[-1][i] = new_loc + prev_ver[i][section_vert.index(new_loc[-1][1]):]
        #                     new_section = False
        #                     break
        #             if new_section:
        #                 prev_ver.append(new_loc)
        #                 cont_sections.append(prev_ver)
                # Perform additional check to connect cont_sections we might have missed
                # This can happen if a new face connects to multiple cont_sections, and we just add it to the first one
                for i in range(len(cont_sections[-1])):
                    # prev_ver[i]'s last vertex == prev_ver[i+1]'s first
                    if cont_sections[-1][i][-1][1] == cont_sections[-1][(i + 1) % len(cont_sections[-1])][0][0]:
                        # Connect!
                        cont_sections[-1][i] += cont_sections[-1][(i + 1) % len(cont_sections[-1])]
                        cont_sections[-1].pop((i + 1) % len(cont_sections[-1]))
                        break
                # cur_boundary = cur_boundary[:index] + new_loc + cur_boundary[index:]
                face = ensure_ccw(face, positions)
                face_list.append(face)
        print(face_list)
        print(cont_sections)
        var = count_non_int_paths_unrestr(face_list, start_edge, start_boundary_list, cont_sections)
        # print("Count for size {0} is {1}".format(size, count_non_int_paths(start_boundary, ('0.0',), dims, 0)))
        print(str(size) + ": " + str(var))
        # print("Duration: " + str(time.time() - start))
        assert var == correct_vals[size] if size < len(correct_vals) else ""



# 185650586758
# 614358697
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
    face_order = [[46,48,130], [46,130,48,128,40], [46,116,47,50], [46,47,116], [46,117,47], [46,40,117], [117,40,47], [128,41,40], [41,131,47,40], [131,52,50,47], [50,52,51], [51,52,106], [52,13,106], [11,13,52,22], [11,22,12], [22,52,23], [23,52,21], [21,52,131,97], [22,23,21], [12,22,35,104], [22,21,35], [35,21,97], [132,128,48], [48,127,132], [127,128,132], [41,128,45], [136,128,127], [126,128,136], [136,127,48], [122,124,136], [136,129,126], [136,124,126,129], [126,45,128], [126,44,45], [124,44,126], [122,43,139,124], [139,43,141,124], [141,43,124], [43,120,124], [120,44,124], [120,42,44], [44,42,45], [42,41,45], [42,121,131,41], [42,131,121], [120,131,42], [43,131,120], [122,131,43], [122,136,48], [122,48,134], [164,122,134,48], [164,131,122], [164,97,131], [164,98,97], [98,86,84,97], [84,35,97], [83,35,84], [85,83,84], [86,85,84], [151,83,85,88], [86,88,85], [82,35,83], [81,35,82], [82,83,53], [81,82,53], [54,81,53], [55,54,53], [53,83,57], [57,83,151], [151,88,29], [29,88,86,31], [87,31,86], [87,96,31], [87,86,98], [164,87,98], [164,96,87], [164,95,96], [96,95,31], [95,93,30,31], [30,29,31], [93,29,30], [55,53,56], [56,53,57], [56,57,58], [58,57,151], [56,58,59], [56,59,60,61], [59,58,60], [60,58,67], [60,67,61], [67,58,151], [67,151,160], [67,160,77], [67,77,160,152], [67,152,160,151,29], [67,29,153], [67,153,68], [153,76,68], [153,29,68,76], [67,68,69], [68,159,69], [68,29,159], [159,29,70], [159,70,69], [61,67,69], [61,69,62], [56,61,62], [55,56,62], [55,62,64], [64,62,63], [64,63,65], [63,62,65], [65,62,144], [62,69,144], [65,144,66], [144,69,66], [66,69,71], [72,66,71], [69,70,71], [164,102,95], [164,103,95,102], [164,89,95,103], [89,94,95], [95,94,93], [164,161,89], [89,161,94], [161,91,93,94], [91,92,93], [92,29,93], [164,90,161], [90,168,161], [90,169,161,168], [169,91,161], [90,91,169], [33,90,164], [33,165,90], [33,166,90,165], [166,91,90], [33,91,166], [33,92,91], [33,114,92], [177,29,92,115], [177,115,92], [114,177,92], [33,34,32,114], [34,112,32], [101,147,177,114], [101,177,147], [170,101,114,32], [37,170,32], [112,37,32], [112,36,37], [38,170,37], [36,38,37], [39,38,36], [112,39,36], [110,39,112], [110,38,39], [110,170,38], [146,101,170], [146,178,101], [146,179,101,178], [146,180,101,179], [146,154,176,177,101,180], [146,176,154], [146,173,177,176], [146,177,173], [155,29,177], [78,29,155], [73,78,155], [156,29,78], [73,156,78], [79,29,156], [73,79,156], [157,29,79], [73,157,79], [80,29,157], [73,80,157], [158,29,80], [73,158,80], [70,29,158], [73,70,158], [71,70,73], [146,73,155,177], [146,75,73], [146,149,75], [146,174,149], [174,182,149], [146,182,174], [146,171,182], [171,172,182], [182,172,149], [172,74,149], [149,74,75], [75,74,73], [74,71,73]]
    enumerate_paths_with_order("data/exp2627neighb.dbf", "data/exp2627wards.shp", face_order, draw=False)
    # enumerate_paths("data/exp2627neighb.dbf", "data/exp2627wards.shp")
    # test()
    # out_dict = {}
    # find_motzkin_paths(0, '', 6, out_dict, 2)
    # print(out_dict)
    exit()
