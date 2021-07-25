import collections
# import modulefinder
import math

import networkx as nx
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import geopandas as gpd
import copy
import time
import itertools
from multiprocessing import Manager, Process

# modulefinder.AddPackagePath()
# import adjacency_graphs

debug = False
fix_end = True


# Create the input grid (as a graph)
class Vertex:
    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def __lt__(self, other):  # An implementation unique ordering of vertices
        x = self.name
        y = other.name
        return [int(x[(x.index('.') + 1):]), -int(x[:x.index('.')])] < [int(y[(y.index('.') + 1):]),
                                                                        -int(y[:y.index('.')])]

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_neighbors(self):
        return self.neighbors

    def remove_neighbor(self, neighbor):
        self.neighbors.remove(neighbor)

    def __str__(self):
        return self.name


# An undirected graph class
class Graph:
    def __init__(self, vertices=None, edges=None):
        if edges is None:
            self.vertices = vertices
            self.edges = []
            self.faces = []
            for vertex in vertices:
                neighb = vertex.get_neighbors()
                self.edges.append(zip([vertex.name, ] * len(neighb), vertex.get_neighbors()))
        elif vertices is None:
            self.edges = edges
            self.vertices = []
            self.faces = []
            for edge in edges:
                self.vertices.append(edge[0]) if edge[0] not in self.vertices else 0
                self.vertices.append(edge[1]) if edge[1] not in self.vertices else 0
        else:
            self.vertices = vertices
            self.edges = edges
            self.faces = []

    def setup_faces(self):
        if len(self.faces) != 0:
            return
        for edge in self.edges:
            for neighbor in edge[1].get_neighbors():
                if edge[0] in neighbor.get_neighbors():
                    face = [edge, (edge[1], neighbor), (neighbor, edge[0])]
                    self.faces.append(face) if face not in self.faces else ""

    def get_vertices(self):
        return self.vertices

    def get_vertex(self, name):
        for v in self.vertices:
            if v.name == name:
                return v
        return None

    def get_edges(self):
        return self.edges

    def get_edge(self, name1, name2):
        for e in self.edges:
            if e[0].name == name1 and e[1].name == name2 or e[0].name == name2 and e[1].name == name1:
                return e
        return None

    def add_edge(self, edge):
        self.edges.append(edge)
        self.vertices.append(edge[0]) if edge[0] not in self.vertices else 0
        self.vertices.append(edge[1]) if edge[1] not in self.vertices else 0
        edge[0].add_neighbor(edge[1])
        edge[1].add_neighbor(edge[0])

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def remove_edge(self, edge):
        self.edges.remove(edge)
        edge[0].remove_neighbor(edge[1])
        edge[1].remove_neighbor(edge[0])

    def remove_vertex(self, vertex=None, vertex_name=None):
        if vertex is None:
            vertex = next((x for x in self.vertices if x.name == vertex_name), None)
        for neighbor in vertex.get_neighbors:
            self.remove_edge([vertex, neighbor])
        self.vertices.remove(vertex)


def setup_grid(width, height):
    vertices = []
    edges = []
    for i in range(width):
        for j in range(height):
            vert = Vertex(str(i) + '.' + str(j))
            if j != 0:
                vert.add_neighbor(vertices[-1])
                vertices[-1].add_neighbor(vert)
                edges.append((vert, vertices[-1]))
            if i != 0:
                vert.add_neighbor(vertices[-height])
                vertices[-height].add_neighbor(vert)
                edges.append((vert, vertices[-height]))
            # if i is not 0 and j is not 0:  # Add an extra edge to triangulate the grid
            #     vert.add_neighbor(vertices[-height - 1])
            #     vertices[-height - 1].add_neighbor(vert)
            #     edges.append((vert, vertices[-height - 1]))
            vertices.append(vert)
    return Graph(vertices, edges)


def number_faces(graph, dims):
    dim = max(dims)
    # Setup the faces:
    face_dict = {}
    for edge in graph.edges:
        face = graph.traverse_face(edge[1], edge[0])  # Traverse clockwise
        if len(face) > 5:  # Hardcode outer face
            continue
        j_arr = sorted([v[(v.index('.') + 1):] for v in face])
        i_arr = sorted([v[:v.index('.')] for v in face])
        ind = i_arr[0] * dim + j_arr[1]
        face_dict[ind] = face
    # graph.setup_faces()
    # for face in graph.faces:
    #     verts = [sorted(edge, key=lambda x: x.name)[0] for edge in face]
    #     # verts = (face[0][0], face[0][1], face[1][1], face[2][1])
    #     j_arr = sorted([v.name[(v.name.index('.') + 1):] for v in verts])
    #     i_arr = sorted([v.name[:v.name.index('.')] for v in verts])
    #     # extra = 0 if int(i_arr[2]) == int(i_arr[1]) else 1
    #     ind = i_arr[0]*dim + j_arr[0]
    #     # ind = int(sorted([v.name[:v.name.index('.')] for v in verts])[1]) * (dim ** 2) + extra * dim + int(j_arr[1])
    #     face_dict[ind] = face
    return face_dict


# Make sure that a face is oriented counterclockwise and starts with the lowest x value
def ensure_ccw(face, positions):
    face_geo = np.asanyarray([positions[x] for x in face])
    min_ind = np.argmin(face_geo, axis=0)[1]  # find the minimum y index
    sgn = np.sign(np.cross(face_geo[(min_ind - 1) % len(face)] - face_geo[min_ind], face_geo[(min_ind + 1) % len(face)]
                           - face_geo[min_ind]))
    min_x_ind = np.argmin(face_geo, axis=0)[0]  # find the minimum x index
    if sgn == 1:
        return [face[(x+min_x_ind) % len(face)] for x in range(len(face))]
    else:
        return [face[(min_x_ind-x) % len(face)] for x in range(len(face))]


class cust_memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, boundary, boundary_labels, face_list, cur_length, exit_edge):
        args = ''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length)
        print(args) if debug else ""
        if args not in self.memo:
            self.memo[args] = self.fn(boundary, boundary_labels, face_list, cur_length, exit_edge)
        print(self.memo) if debug else ""
        return self.memo[args]


def proc_helper(memo, argstring, fn, args):
    memo[argstring] = fn(*args)


job_manager = Manager()
job_list = job_manager.list()


class cust_memoize_no_length:
    def __init__(self, fn):
        self.fn = fn
        memo_manager = Manager()
        self.memo = memo_manager.dict()

    def __call__(self, boundary, boundary_labels, face_list, cur_length, exit_edge):
        if math.floor(time.time() * 10000000) % 100000 == 11111:  # Stop randomly about 1/10000 times
            print("Have processed {0} entries so far.".format(len(self.memo)))
        args = ''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list))
        print(args) if debug else ""
        if args not in self.memo:
            p = Process(target=proc_helper, args=(self.memo, args, self.fn, (boundary, boundary_labels, face_list, cur_length, exit_edge)))
            job_list.append(p)
            p.start()
            # self.memo[args] = self.fn(boundary, boundary_labels, face_list, cur_length, exit_edge)
        print(self.memo) if debug else ""
        return self.memo[args]


# The main recursive method to populate the memoization table with counts of nonintersecting paths
# Args:
# boundary - a list with edges corresponding to the boundary labels
# boundary_labels - a list with the labels corresponding to the boundary
# face_list - A list of the faces left to step over. Only the length is memoized
# cur_length - Not always memoized, the current length of the path
# exit_edge - Not memoized, the edge of the dual to exit on
@cust_memoize_no_length
def count_non_int_paths(boundary, boundary_labels, face_list, cur_length, exit_edge):
    # print([(k[0].name, k[1].name) for k in boundary.keys()]) if debug else ""
    # if len(face_list) < 10:
    #     pass
    print(boundary) if debug else ""
    print(boundary_labels) if debug else ""
    for i in range(len(boundary) - 1):
        if len(set(boundary[i]).intersection(set(boundary[i + 1]))) != 1:
            # pass
            raise Exception("Boundary is not contiguous")
    # if cur_length > 3*(size-1):
    #     return 0
    # if cur_length + np.floor(len(face_list)/(size-1)) > 3*(size-1):
    #     return 0
    if len(face_list) == 0:  # Already technically outside, only need to check if this ending configuration is valid
        if 2 in boundary_labels or 3 in boundary_labels:
            # We have a separate component
            return 0
        elif 1 in boundary_labels:
            # Terminate a free end that reaches the other side
            if fix_end:
                end_edge = tuple(sorted(boundary[boundary_labels.index(1)]))
                if exit_edge == end_edge:
                    return 1
                else:
                    return 0
            else:
                return 1
        else:
            raise Exception("Ended with an invalid boundary")
    face = face_list.pop(0)
    # stores the current labels
    labels = []
    # stores the edges in the boundary
    # cur_edges = []
    # stores the new vertices that will be added to the boundary
    new_loc = []
    # stores the index all face elements will be put to
    index = len(boundary)
    for i in range(len(face)):
        edge = (face[i], face[((i + 1) % len(face))])
        named_edge = tuple(sorted(edge))
        if named_edge in boundary:
            cur_index = boundary.index(named_edge)
            lab = boundary_labels[cur_index]
            if lab != 0:
                labels.append(lab)
            boundary.pop(cur_index)
            boundary_labels.pop(cur_index)
            if cur_index < index:
                index = cur_index
        else:
            new_loc.append(named_edge)
    # Make sure new_loc follows the order of boundary:
    swapped = len(boundary) > 0 and len(new_loc) > 0 and \
              (len(boundary) > index > 0 and (len(set(new_loc[0]).intersection(set(boundary[index - 1]))) == 0 or len(
                  set(new_loc[-1]).intersection(set(boundary[index]))) == 0))
    if swapped:  # need to find where to start the new locations
        start_ind = -1
        for i in range(len(new_loc)):
            if len(set(new_loc[i]).intersection(set(boundary[index - 1]))) > 0:
                start_ind = i
                if len(set(new_loc[i]).intersection(set(boundary[index - 1]))) == 1 and len(
                  set(new_loc[(i-1) % len(new_loc)]).intersection(set(boundary[index]))) == 1:
                    print("Found good rotation: " + str(new_loc)) if debug else ""
                    break
        new_loc = [new_loc[(x + start_ind) % len(new_loc)] for x in range(len(new_loc))]
    labels = reversed(labels)
    labels = tuple(labels)
    new_loc = tuple(new_loc)
    if len(labels) > 2:  # We reached a case where at least three paths meet
        return 0
    print("Current face: " + str(face)) if debug else ""
    print("New location: " + str(new_loc)) if debug else ""
    print("Labels: " + str(labels)) if debug else ""
    if len(labels) == 0:
        total_count = 0
        boundary1 = copy.deepcopy(boundary)
        boundary_labels1 = copy.deepcopy(boundary_labels)
        offset = 0
        for loc in new_loc:
            boundary1 = boundary1[:index + offset] + [loc] + boundary1[index + offset:]
            boundary_labels1 = boundary_labels1[:index + offset] + [0] + boundary_labels1[index + offset:]
            offset += 1
        total_count += count_non_int_paths(boundary1, boundary_labels1, copy.deepcopy(face_list), cur_length, exit_edge)
        for ind1, ind2 in itertools.combinations(range(len(new_loc)), 2):  # itertools.combinations preserves order!
            boundary1 = copy.deepcopy(boundary)
            boundary_labels1 = copy.deepcopy(boundary_labels)
            offset = 0
            for i in range(len(new_loc)):
                boundary1 = boundary1[:index+offset] + [new_loc[i]] + boundary1[index + offset:]
                boundary_labels1 = boundary_labels1[:index+offset] + [3 if i==ind1 else 2 if i==ind2 else 0] + boundary_labels1[index+offset:]
                offset += 1
            total_count += count_non_int_paths(boundary1, boundary_labels1, copy.deepcopy(face_list), cur_length + 2,
                                               exit_edge)
        del boundary
        del boundary_labels
        return total_count
    elif labels == (1,):
        total_count = 0
        for loc1 in new_loc:
            boundary2 = copy.deepcopy(boundary)
            boundary_labels2 = copy.deepcopy(boundary_labels)
            offset = 0
            for loc2 in new_loc:
                boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
                boundary_labels2 = boundary_labels2[:index + offset] + [1 if loc2 == loc1 else 0] + boundary_labels2[
                                                                                                    index + offset:]
                offset += 1
            total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list), cur_length + 1,
                                               exit_edge)
        del boundary
        del boundary_labels
        return total_count
    elif labels == (2,):
        total_count = 0
        for loc1 in new_loc:
            boundary2 = copy.deepcopy(boundary)
            boundary_labels2 = copy.deepcopy(boundary_labels)
            offset = 0
            for loc2 in new_loc:
                boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
                boundary_labels2 = boundary_labels2[:index + offset] + [2 if loc2 == loc1 else 0] + boundary_labels2[
                                                                                                    index + offset:]
                offset += 1
            total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list), cur_length + 1,
                                               exit_edge)
        del boundary
        del boundary_labels
        return total_count
    elif labels == (3,):
        total_count = 0
        for loc1 in new_loc:
            boundary2 = copy.deepcopy(boundary)
            boundary_labels2 = copy.deepcopy(boundary_labels)
            offset = 0
            for loc2 in new_loc:
                boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
                boundary_labels2 = boundary_labels2[:index + offset] + [3 if loc2 == loc1 else 0] + boundary_labels2[
                                                                                                    index + offset:]
                offset += 1
            total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list), cur_length + 1,
                                               exit_edge)
        del boundary
        del boundary_labels
        return total_count
    elif labels in [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (3, 3), (2, 3)]:
        boundary1 = copy.deepcopy(boundary)
        boundary_labels1 = copy.deepcopy(boundary_labels)
        offset = 0
        for loc in new_loc:
            boundary1 = boundary1[:index + offset] + [loc] + boundary1[index + offset:]
            boundary_labels1 = boundary_labels1[:index + offset] + [0] + boundary_labels1[index + offset:]
            offset += 1
        del boundary
        if labels == (2, 3):  # possible, just combine
            # return 0
            return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
        # Need to find partner and change label:
        if 3 in labels:  # 2 will be below it
            # keys_in_order = keys_in_order[keys_in_order.index(new_loc[0]):]
            count = 0
            for i in range(index, len(boundary1)):
                if boundary_labels1[i] == 3:
                    count += 1
                if boundary_labels1[i] == 2:
                    if count != 0:
                        count -= 1
                    else:
                        boundary_labels1[i] = 1 if labels != (3, 3) else 3
                        return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
        else:
            count = 0
            for i in range(index, 0, -1):
                if boundary_labels1[i] == 2:
                    count += 1
                if boundary_labels1[i] == 3:
                    if count != 0:
                        count -= 1
                    else:
                        boundary_labels1[i] = 1 if labels != (2, 2) else 2
                        return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
        raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")
    elif labels == (3, 2):
        # print("Closed a loop!")
        # print(''.join([str(x) for x in boundary_labels]) + "." + str(len(face_list)) + "." + str(cur_length))
        return 0
        # We just closed a loop! Currently allowed
        raise Exception("Theoretically impossible case occurred, we closed a loop.")
    else:
        raise Exception("Invalid labels on step location")


# Method to enumerate all possible non-self-intersecting paths crossing a given shapefile and adjacency information
# corresponding to the shapefile.
# Args:
# adj_file - the path to a dbf file generated from arcgis's polygon neighborhood tool that contains the adjacency
# information. NOTE: The graph corresponding to this data should be planar, but may have dangling vertices.
# shapefile - the path to a shp file that contains the local geometry and whatever voter data. For now, this is only
# used to compute centroids which ensure proper orientation for the planar graph.
def enumerate_paths(adj_file, shapefile, recalculate=False, draw=True):
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
    y_locs = {x: centers.loc[x]['centroid_column'].y for x in h.nodes}
    stddev = np.std(np.asanyarray(list(y_locs.values())))
    center = np.mean(np.asanyarray([y_locs[exit_edge[0]], y_locs[exit_edge[1]], y_locs[start_edge[0]], y_locs[start_edge[0]]]))
    new_verts = [x for x in h.nodes if math.fabs(y_locs[x]-center) < stddev/2.25]
    h2 = h.subgraph(new_verts).copy()
    g_data = {x: [y for y in g_data[x] if y in new_verts] for x in new_verts}
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
            verts_to_clean = verts_to_clean.union(set(face[i1+1:i2]))
            points_to_keep.add(point)
            face_to_add = face[:i1] + face[i2:]
            face_dict[str(sorted(face_to_add))] = face_to_add
        else:
            face_dict[str(sorted(face))] = face
    # Remove boundary loops
    while len(np.unique(np.asanyarray(start_boundary_list).flatten()))*2-2 != len(np.asanyarray(start_boundary_list).flatten()):
        flat_boundary = np.asanyarray(start_boundary_list).flatten()
        unique, indices, counts = np.unique(flat_boundary, return_counts=True, return_index=True)
        point_ind = np.argmax(counts)
        point = unique[point_ind]
        i1 = indices[point_ind]
        i2 = np.where(flat_boundary == point)[0][2]
        verts_to_clean = verts_to_clean.union(set(flat_boundary[i1 + 1:i2]))
        points_to_keep.add(point)
        start_boundary_list = start_boundary_list[:int(np.floor(i1/2)+1)] + start_boundary_list[int(np.floor(i2/2)+1):]
        start_boundary_labels = start_boundary_labels[:int(np.floor(i1/2)+1)] + start_boundary_labels[int(np.floor(i2/2)+1):]
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
    # face_list = sorted(face_dict.values(), key=lambda face: sorted([positions[x][0] for x in face]))
    # face_list = sorted(face_dict.values(), key=lambda face: np.mean([positions[x][0] for x in face]))
    # Ensure that face_list results in a continuous boundary
    face_list = []
    cur_edge_index = start_boundary_list.index(start_edge)
    cur_edge_index = start_boundary_list.index((start_edge[1], start_edge[0])) if cur_edge_index == -1 else cur_edge_index
    cur_boundary = copy.deepcopy(start_boundary_list)
    while len(cur_boundary) > 1:
        cur_edge = cur_boundary[cur_edge_index]
        face = graph.traverse_face(cur_edge[1], cur_edge[0])
        # flat_boundary = np.asanyarray(cur_boundary).flatten()
        boundary_verts = [x[0] for x in cur_boundary] + [cur_boundary[-1][1]]
        # Check if frontier still simple:
        inds = []
        for vertex in face:
            # ind_list = np.where(flat_boundary == vertex)[0]
            # if len(ind_list) > 0:
            #     inds.append(np.floor(ind_list[0]/2))
            try:
                ind = boundary_verts.index(vertex)
                inds.append(ind)
            except ValueError:
                pass
        cont = False
        inds = sorted(inds)
        for i in range(len(inds)-1):
            if inds[i+1]-inds[i] > 1:  # Bad face! Go to next? edge
                cur_edge_index = (cur_edge_index + 1) % len(cur_boundary)
                cont = True
        if cont:
            continue
        # Face is good! Add to boundary
        # stores the new vertices that will be added to the boundary
        new_loc = []
        # stores the index all face elements will be put to
        index = len(cur_boundary)
        for i in range(len(face)):
            edge = (face[((i + 1) % len(face))], face[i])  # We know it'll be reversed!
            # named_edge = tuple(sorted(edge))
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
        cur_boundary = cur_boundary[:index] + new_loc + cur_boundary[index:]
        face = ensure_ccw(face, positions)
        face_list.append(face)
    start_boundary_list = [tuple(sorted(x)) for x in start_boundary_list]
    print(face_list)
    return count_non_int_paths(start_boundary_list, start_boundary_labels, face_list, 0, exit_edge)


# def sample(memo_dict, boundary, face_list):
#     start_key = max([v for v in memo_dict.values()], key=lambda x: memo_dict[x])
#     cur_key = start_key
#     while int(cur_key[cur_key.index('.')+1:cur_key.index('.', start=cur_key.index('.')+1)]) != 0:
#         face = face_list.pop(0)
#         new_loc, labels = [], []
#         index = len(boundary)
#         for i in range(len(face)):
#             edge = (face[i], face[((i + 1) % len(face))])
#             named_edge = tuple(sorted(edge))
#             if named_edge in boundary:
#                 cur_index = boundary.index(named_edge)
#                 lab = cur_key[cur_index]
#                 if lab is not 0:
#                     labels.append(lab)
#                     # cur_edges.append(named_edge)
#                 boundary.pop(cur_index)
#                 cur_key.pop(cur_index)
#                 if cur_index < index:
#                     index = cur_index
#             else:
#                 new_loc.append(named_edge)
#         labels = reversed(labels)
#         labels = tuple(labels)
#         new_loc = tuple(new_loc)
#         counts = []
#         if len(labels) == 0:
#             total_count = 0
#             boundary1 = copy.deepcopy(boundary)
#             boundary_labels1 = copy.deepcopy(cur_key)
#             offset = 0
#             for loc in new_loc:
#                 boundary1 = boundary1[:index + offset] + [loc] + boundary1[index + offset:]
#                 boundary_labels1 = boundary_labels1[:index + offset] + [0] + boundary_labels1[index + offset:]
#                 offset += 1
#             counts.append(memo_dict[boundary_labels1])
#             for loc1, loc2 in itertools.combinations(new_loc, 2):  # itertools.combinations preserves order!
#                 boundary1 = copy.deepcopy(boundary)
#                 boundary_labels1 = copy.deepcopy(boundary_labels)
#                 boundary1 = boundary1[:index] + [loc1] + boundary1[index:]
#                 boundary_labels1 = boundary_labels1[:index] + [3] + boundary_labels1[index:]
#                 boundary1 = boundary1[:index + 1] + [loc2] + boundary1[index + 1:]
#                 boundary_labels1 = boundary_labels1[:index + 1] + [2] + boundary_labels1[index + 1:]
#                 total_count += count_non_int_paths(boundary1, boundary_labels1, copy.deepcopy(face_list),
#                                                    cur_length + 2,
#                                                    exit_edge)
#             del boundary
#             del boundary_labels
#             return total_count
#         elif labels == (1,):
#             total_count = 0
#             for loc1 in new_loc:
#                 boundary2 = copy.deepcopy(boundary)
#                 boundary_labels2 = copy.deepcopy(boundary_labels)
#                 offset = 0
#                 for loc2 in new_loc:
#                     boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
#                     boundary_labels2 = boundary_labels2[:index + offset] + [
#                         1 if loc2 == loc1 else 0] + boundary_labels2[
#                                                     index + offset:]
#                     offset += 1
#                 total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list),
#                                                    cur_length + 2,
#                                                    exit_edge)
#             del boundary
#             del boundary_labels
#             return total_count
#         elif labels == (2,):
#             total_count = 0
#             for loc1 in new_loc:
#                 boundary2 = copy.deepcopy(boundary)
#                 boundary_labels2 = copy.deepcopy(boundary_labels)
#                 offset = 0
#                 for loc2 in new_loc:
#                     boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
#                     boundary_labels2 = boundary_labels2[:index + offset] + [
#                         2 if loc2 == loc1 else 0] + boundary_labels2[
#                                                     index + offset:]
#                     offset += 1
#                 total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list),
#                                                    cur_length + 2,
#                                                    exit_edge)
#             del boundary
#             del boundary_labels
#             return total_count
#         elif labels == (3,):
#             total_count = 0
#             for loc1 in new_loc:
#                 boundary2 = copy.deepcopy(boundary)
#                 boundary_labels2 = copy.deepcopy(boundary_labels)
#                 offset = 0
#                 for loc2 in new_loc:
#                     boundary2 = boundary2[:index + offset] + [loc2] + boundary2[index + offset:]
#                     boundary_labels2 = boundary_labels2[:index + offset] + [
#                         3 if loc2 == loc1 else 0] + boundary_labels2[
#                                                     index + offset:]
#                     offset += 1
#                 total_count += count_non_int_paths(boundary2, boundary_labels2, copy.deepcopy(face_list),
#                                                    cur_length + 2,
#                                                    exit_edge)
#             del boundary
#             del boundary_labels
#             return total_count
#         elif labels in [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (3, 3), (2, 3)]:
#             boundary1 = copy.deepcopy(boundary)
#             boundary_labels1 = copy.deepcopy(boundary_labels)
#             offset = 0
#             for loc in new_loc:
#                 boundary1 = boundary1[:index + offset] + [loc] + boundary1[index + offset:]
#                 boundary_labels1 = boundary_labels1[:index + offset] + [0] + boundary_labels1[index + offset:]
#                 offset += 1
#             del boundary
#             if labels == (2, 3):  # possible, just combine
#                 # return 0
#                 return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
#             # Need to find partner and change label:
#             if 3 in labels:  # 2 will be below it
#                 # keys_in_order = keys_in_order[keys_in_order.index(new_loc[0]):]
#                 count = 0
#                 for i in range(index, len(boundary1)):
#                     if boundary_labels1[i] == 3:
#                         count += 1
#                     if boundary_labels1[i] == 2:
#                         if count != 0:
#                             count -= 1
#                         else:
#                             boundary_labels1[i] = 1 if labels != (3, 3) else 3
#                             return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
#             else:
#                 count = 0
#                 for i in range(index, 0, -1):
#                     if boundary_labels1[i] == 2:
#                         count += 1
#                     if boundary_labels1[i] == 3:
#                         if count != 0:
#                             count -= 1
#                         else:
#                             boundary_labels1[i] = 1 if labels != (2, 2) else 2
#                             return count_non_int_paths(boundary1, boundary_labels1, face_list, cur_length, exit_edge)
#             raise Exception("Failed to match a 3 to a 2 or a 2 to 3.")


# A quick testing framework
def test():
    correct_vals = [0, 0, 0, 2, 12, 184, 8512, 1262816, 575780564, 789360053252, 3266598486981642]
    global size
    for size in [4, 5, 6]:
        grid = setup_grid(size, size)
        grid_graph = nx.PlanarEmbedding()
        grid_graph.set_data({v.name: [n.name for n in reversed(v.neighbors)] for v in grid.vertices})
        # plt.subplot(111)
        # nx.draw(grid_graph)
        # plt.show()
        face_dict = number_faces(grid_graph, (size, size))
        face_list = [face_dict[k] for k in sorted(face_dict.keys())]
        exit_edge = (str(size - 1) + '.' + str(size - 2), str(size - 1) + '.' + str(size - 1))
        outer_face = max([grid_graph.traverse_face(*exit_edge), grid_graph.traverse_face(exit_edge[1], exit_edge[0])],
                         key=lambda x: len(x))
        start_boundary_list = []
        start_boundary_labels = []
        for i in range(len(outer_face)):
            edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
            if edge == exit_edge:
                continue
            start_boundary_list.append(edge)
            start_boundary_labels.append(1 if edge == ('0.0', '0.1') else 0)

        start = time.time()
        var = count_non_int_paths(start_boundary_list, start_boundary_labels, face_list, 0, exit_edge)
        # print("Count for size {0} is {1}".format(size, count_non_int_paths(start_boundary, ('0.0',), dims, 0)))
        # print(str(size) + ": " + str(var))
        # print("Duration: " + str(time.time() - start))
        assert var == correct_vals[size] if size < len(correct_vals) else ""
        intermediate_counts = np.zeros(size - 1)
        intermediate_counts2 = np.zeros(size - 1)
        for key, val in count_non_int_paths.memo.items():
            cur_size = int(key[key.index(".") + 1:key.index(".", key.index(".") + 1)])
            if cur_size % (size - 1) == 0:
                intermediate_counts[int((cur_size / (size - 1)) - 1)] += 1
                intermediate_counts2[int((cur_size / (size - 1)) - 1)] += 1 if val == 0 else 0
        # print([x for x in reversed(intermediate_counts)])
        # print([x for x in reversed(intermediate_counts2)])
        # print(list(count_non_int_paths.memo.values()).count(0))
        # assert count_non_int_paths(start_boundary_list, start_boundary_labels, face_list, 0, exit_edge) == correct_vals[
        #     size]
        count_non_int_paths.memo = {}


# test()
enumerate_paths("data/exp2627neighb.dbf", "data/exp2627wards.shp")
for job in job_list:
    job.join()
exit(0)

size = 4
# These are useless, but compiler needs help
zeros_due_to_smart_length = 0
zeros_due_to_length = 0

grid = setup_grid(size, size)
grid_graph = nx.PlanarEmbedding()
grid_data = {v.name: [n.name for n in reversed(v.neighbors)] for v in grid.vertices}
# grid_data['1.0'].remove('1.1')
# grid_data['1.1'].remove('1.0')
grid_data['0.0'] = grid_data['0.0'][:1] + ['1.1'] + grid_data['0.0'][1:]
grid_data['1.1'] = grid_data['1.1'][:3] + ['0.0'] + grid_data['1.1'][3:]
print(grid_data)
grid_graph.set_data(grid_data)
plt.subplot(111)
nx.draw(grid_graph)
plt.show()

# A note on grid dimensions: in a 5x5 grid, we will say there are 4x4 squares.
# First try a 5x5 grid example:
face_dict = number_faces(grid_graph, (size, size))
face_list = [face_dict[k] for k in sorted(face_dict.keys())]
# print([[[v.name for v in e] for e in f] for f in face_list])
# start_edge_names = [('0.' + str(i), '0.' + str(i + 1)) for i in range(size) if i != size - 1]
# # start_edges = [tuple(sorted((edge[0].name, edge[1].name))) for edge in grid.edges if
# #                (edge[0].name, edge[1].name) in start_edge_names or (edge[1].name, edge[0].name) in start_edge_names]
# start_boundary = {
#     edge: 1 if (edge[0], edge[1]) == ('0.0', '0.1') or (edge[0], edge[1]) == ('0.1', '0.0') else 0
#     for edge in start_edge_names}
exit_edge = (str(size - 1) + '.' + str(size - 2), str(size - 1) + '.' + str(size - 1))
outer_face = max([grid_graph.traverse_face(*exit_edge), grid_graph.traverse_face(exit_edge[1], exit_edge[0])],
                 key=lambda x: len(x))
start_boundary_list = []
start_boundary_labels = []
for i in range(len(outer_face)):
    edge = tuple(sorted([outer_face[i], outer_face[(i + 1) % len(outer_face)]]))
    if edge == exit_edge:
        continue
    start_boundary_list.append(edge)
    start_boundary_labels.append(1 if edge == ('0.0', '0.1') else 0)
# add weird faces to face_list
face_list = [['0.0', '1.1', '0.1']] + face_list
print(face_list)
print(count_non_int_paths(start_boundary_list, start_boundary_labels, face_list, 0, exit_edge))
print(count_non_int_paths.memo)
# start_boundary = {'0.0': 0, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 1}
# for size in [11, 12]:
#     dims = (size, size)
#     start_boundary = {}
#     for i in range(size):
#         start_boundary['0.' + str(i)] = 0 if i != size-1 else 1
#     start = time.time()
#     print("Count for size {0} is {1}".format(size, count_non_int_paths(start_boundary, ('0.0',), dims, 0)))
#     print("Duration: " + str(time.time() - start))


# print(count_non_int_paths({'0.0': 1, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0}, ('0.0',), (5, 5), 0) + count_non_int_paths({'0.0': 0, '0.1': 1, '0.2': 0, '0.3': 0, '0.4': 0}, ('0.0',), (5, 5), 0) + count_non_int_paths({'0.0': 0, '0.1': 0, '0.2': 1, '0.3': 0, '0.4': 0}, ('0.0',), (5, 5), 0) + count_non_int_paths({'0.0': 0, '0.1': 0, '0.2': 0, '0.3': 1, '0.4': 0}, ('0.0',), (5, 5), 0) + count_non_int_paths({'0.0': 0, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 1}, ('0.0',), (5, 5), 0))
# grid = setup_grid(2, 2)
# print(count_non_int_paths({'0.0': 1, '0.1': 0}, ('0.0', ), (2, 2), 0))
