# Core idea: Enumerate Motzkin paths into preallocated data structure
# Each contiguous section of the frontier has its own motzkin paths
# The 1 could be anywhere in the start_edge's contiguous section
# Maintain a dictionary of edge relations
# Motzkin paths should be enumerated by putting in either a 0 or a 3 or 2 if 3 is before
# List of dicts?
# Only main algorithm is bottom up, rest can be top down
# 33179817984000

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

debug = True


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
                    for start_depth in range(5):  # will repeat, but repeats are ok
                        find_motzkin_paths(0, '', len(section) - one_ind - 1, first_dict, start_depth)
                        find_motzkin_paths(0, '', one_ind, second_dict, 4 - start_depth)
                        my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                cur_dict = {s1 + s2: 0 for s1 in cur_dict.keys() for s2 in my_dict.keys()}
            else:
                my_dict = {}
                find_motzkin_paths(0, '', len(section), my_dict, 0)
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


def count_non_int_paths_restr_length(face_list, start_edge, outer_boundary, cont_sections, cutoff_length):
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
    # Keep track of the minimum and maximum lengths to set up for the memoization table
    max_length = 1
    min_length = 1
    for i in range(2, len(cont_sections)+1):  # Range is off because negative values are offset
        cur_sect = cont_sections[-i]
        cur_dict = collections.defaultdict()
        cur_dict[''] = 0
        # gc.collect()
        for length in range(min_length, max_length + 2):
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
                        find_motzkin_paths(0, '', len(section) - one_ind - 1, first_dict)
                        find_motzkin_paths(0, '', one_ind, second_dict)
                        my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                    cur_dict = {str(length) + '.' + s1 + s2: 0 for s1 in cur_dict.keys() for s2 in my_dict.keys()}
                else:
                    my_dict = {}
                    find_motzkin_paths(0, '', len(section), my_dict)
                    cur_dict = {str(length) + '.' + s1 + s2: 0 for s1 in cur_dict.keys() for s2 in my_dict.keys()}
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
def find_motzkin_paths(h, w, n, m_dict, depth):
    j = len(w)
    if depth > 5:
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
                    find_motzkin_paths(0, '', len(section) - i, first_dict)
                    find_motzkin_paths(0, '', i, second_dict)
                    my_dict.update({s1 + '1' + s2: 0 for s1 in first_dict.keys() for s2 in second_dict.keys()})
                path_dict = {s1 + s2: 0 for s1 in path_dict.keys() for s2 in my_dict.keys()}
            else:
                my_dict = {}
                find_motzkin_paths(0, '', len(section), my_dict)
                path_dict = {s1 + s2: 0 for s1 in path_dict.keys() for s2 in my_dict.keys()}
        big_table.append(path_dict)
    # Generate edge relations
    edge_dict = collections.defaultdict()
    for i in range(len(cont_sections) - 1):
        face = face_list[i]
        label_inds = []
        new_loc = []
        # Find index of step
        flattened_sections = [x for x in cont_sections[i][j] for j in range(len(cont_sections[i]))]
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
        for path in cont_sections[i].keys():
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
    # print(np_df)
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
        nx.draw(g, pos=positions, node_size=30, with_labels=True, font_size=6, font_color='red')
        # G2 = h.subgraph([106,0,3,4,5,6,7,8,9,14,107,13,51,52])
        # nx.draw(G2, pos=positions, with_labels=True)
        # centers.plot()
        # loc_df.plot()
        plt.show()
    g.check_structure()
    print(order_faces(g, positions))


def order_faces(graph, positions):
    # mat = nx.adjacency_matrix(graph, nodelist=list(range(158))).toarray()
    # for row in mat:
    #     for x in row:
    #         print(str(x)+', ', end='')
    #     print()
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
    graph.check_structure()
    # h = nx.DiGraph(graph.get_data())
    # G2 = h.subgraph(outer_face)
    # nx.draw(G2, pos=positions, with_labels=True, node_size=30)
    plt.show()
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
    # mat = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes)).toarray()
    # for row in mat:
    #     for x in row:
    #         print(str(int(x/2))+', ', end='')
    #     print()
    # exit()

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
                cur_edge_index = (cur_edge_index + 5) % (len(cur_boundary) - 1) if pass_counter < len(outer_face)*20 else \
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
            sect_mem *= m_n[len(sect)] if len(sect) < len(m_n) else m_n[-1]
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


# 185650586758
# 614358697
if __name__ == '__main__':
    # m_n = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835, 113634, 310572, 853467, 2356779, 6536382,
    #        18199284, 50852019, 142547559, 400763223, 1129760415, 3192727797, 9043402501, 25669818476, 73007772802,
    #        208023278209, 593742784829]
    # n = 29
    # d = 5
    # for d in range(math.ceil(n/2)):
    #     print("Number of paths w/ depth <= {0}: {1}".format(d, m_n[n] - sum([1/(d2+1) * (math.comb(2*d2, d2) * math.comb(n, 2*d2)) for d2 in range(d, math.floor(n/2)+1)])))
    # for n in range(10, 30):
    #     print("Number of paths of length {0} w/ depth <= {1}: {2}".format(n, d, m_n[n] - sum([1/(d2+1) * (math.comb(2*d2, d2) * math.comb(n, 2*d2)) for d2 in range(d, math.floor(n/2)+1)])))
    # for d in range(math.ceil(n/2)):
    #     print("Number of paths w/ depth = {0}: {1}".format(d, 1/(d+1) * (math.comb(2*d, d) * math.comb(n, 2*d))))
    enumerate_paths("data/exp2627neighb.dbf", "data/exp2627wards.shp")
    exit()
