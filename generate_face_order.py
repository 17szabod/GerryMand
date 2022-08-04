import collections
import copy
import gc
import itertools
import math
import time
from copyreg import remove_extension
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import geopandas as gpd
import pandas
from shapely.geometry import Polygon, Point
import non_int_bottom_up


# Makes a new subgraph from a nx graph induced by nodes
def make_subgraph(g, nodes):
    # Create a subgraph SG based on a (possibly multigraph) G
    sg = nx.PlanarEmbedding()
    # sg.add_nodes_from((n, g.nodes[n]) for n in nodes)  # keeps attributes
    sg.add_nodes_from(nodes)  # doesn't keep attributes
    data = g.get_data()
    new_data = dict()
    for n in nodes:
        new_data[n] = [v for v in data[n] if v in nodes and (n,v) in g.edges]
    sg.set_data(new_data)
    success = nx.check_planarity(sg)
    if not success:
        raise Exception("Not planar")
    return sg


# nx planar embedding does not fix adjacency information when edges are removed, need to do it manually
def remove_planar_edges(g, edges):
    g_data = g.get_data()
    for edge in edges:
        g_data[edge[0]].remove(edge[1])
        g_data[edge[1]].remove(edge[0])
    g2 = nx.PlanarEmbedding()
    g2.set_data(g_data)
    return g2


class memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, verts, face, positions, geom_dict, use_graph=True):
        args = str(verts) + '.' + str(face)
        if args not in self.memo:
            self.memo[args] = self.fn(verts, face, positions, geom_dict, use_graph)
        return self.memo[args]


# Uses shapely to test if a vertex is contained in a face
# @param verts: a single vertex or a sequence of vertices to test
# @param face: an ordered list of vertices of the face
# @param positions: the dictionary of positions
# @param use_graph: a boolean saying whether vertices are given as graph vertices or positions are given
@memoize
def vert_in_face(verts, face, positions, geom_dict, use_graph=True):
    one_vert = False
    if use_graph:
        if not hasattr(verts, '__len__'):
            one_vert = True
            verts = [verts]
        points = pandas.Series([geom_dict[v] for v in verts])
    else:
        if type(verts[0]) is not type(positions[face[0]]):  # verts is just a single position
            one_vert = True
            verts = [verts]
        points = pandas.Series([Point(v) for v in verts])
    poly = Polygon([geom_dict[x] for x in face])
    to_ret = list(points.map(lambda p: poly.distance(p) <= .000001))
    return to_ret[0] if one_vert else to_ret


def vert_in_face_mtpltlib(v, face, positions):
    if any([x == v for x in face]):
        return True
    path = mpath.Path(list([positions[x] for x in face]))
    return path.contains_point(positions[v])


# Given a planar graph g, generate a face order starting with edge
# @param g_copy: The input graph g. this will be copied into a new graph that can change, but we still need to access it
# @param edge: The edge to start the traversal with
# @param positions: The position dictionary of every node in g and maybe more
# @param bottom: The 'safe edges' that are part of the traversal order, sent as a dictionary of edges
# Presupposes that bottom is a subset of the outer face of g_copy
def order_faces(g_copy, edge, positions, geom_dict, bottom, exit_edge=None):
    bottom_verts = set()
    for b in bottom:
        bottom_verts.add(b[0])
        bottom_verts.add(b[1])
    g = copy.deepcopy(g_copy)  # The graph to be modified in each iteration
    g_prime = copy.deepcopy(g)  # The version of g with outer edges removed
    if edge is None:  # Algorithm couldn't find a good edge, so just choose one
        edge = min([e for e in g.edges if all(vert_in_face(g.nodes, g.traverse_face(*e), positions, geom_dict))], key=lambda e: len(g.traverse_face(*e)))
    # outer_face = max([g.traverse_face(*edge), g.traverse_face(edge[1], edge[0])], key=lambda x: len(x))
    outer_face = min([g.traverse_face(*edge), g.traverse_face(edge[1], edge[0])], key=lambda x: len(x) if all(vert_in_face(g.nodes, x, positions, geom_dict)) else np.infty)
    if not all(vert_in_face(g.nodes, outer_face, positions, geom_dict)):
        raise BaseException("Whats up with this face?")
    outer_face = outer_face[1:] + [outer_face[0]]
    outer_face_edges = [(outer_face[i], outer_face[(i + 1) % len(outer_face)]) for i in range(len(outer_face))]
    rev_outer_face_edges = [(outer_face[(i + 1) % len(outer_face)], outer_face[i]) for i in range(len(outer_face))]
    if not all([b in outer_face_edges or b in rev_outer_face_edges for b in bottom]):
        bottom = {b for b in bottom if b in g.edges}  # No point keeping those that don't exist
    if nx.has_bridges(g.to_undirected(reciprocal=True)):
        # Simply remove these edges and induct on each nontrivial component
        bridges = list(nx.bridges(g.to_undirected(reciprocal=True)))
        g = remove_planar_edges(g, bridges)
        new_graphs = nx.connected_components(g)
        local_traversal = []
        for coco in new_graphs:
            if len(coco) > 1:
                g2 = make_subgraph(g, coco)
                new_bottom = set((b for b in bottom if b in g2.edges))
                new_edge = edge if edge in g2.edges else ""
                if new_edge == "":
                    for safe_edge in new_bottom:
                        if any(b[0] in safe_edge or b[1] in safe_edge for b in bridges):
                            new_edge = safe_edge
                local_traversal += order_faces(g2, new_edge, positions, geom_dict, new_bottom)
        return local_traversal

    g_prime = remove_planar_edges(g_prime, outer_face_edges)
    if g_prime.size() == 0:
        if nx.node_connectivity(g) == 1:
            # We have bridge nodes
            for v in nx.all_node_cuts(g.to_undirected(reciprocal=True), k=1):  # Should ideally not happen
                v = v.pop()  # Gives as a set
                g2 = copy.deepcopy(g)
                g2.remove_node(v)
                cocos = list(nx.connected_components(g2))
                return [list(cocos[0].union({v})), list(cocos[1].union({v}))] if edge[0] in cocos[0].union({v}) and edge[1] in cocos[0].union({v}) else [list(cocos[1].union({v})), list(cocos[0].union({v}))]
        # Base case, only possible with a single face.
        while min(g.out_degree, key=lambda x: x[1])[1] < 2:  # bridge edges are now handled above
            g.remove_node(min(g.out_degree, key=lambda x: x[1])[0])
        # inner_face = max([min([g.traverse_face(*e), g.traverse_face(e[1], e[0])], key=lambda x: len(x)) for e in g.edges()])
        return [list(g.nodes)]

    if len(outer_face) == 3:  # graphs embedded in a triangle are a special case, just remove one face and induct
        return step_one_face(edge, g, outer_face, positions, geom_dict, bottom)
    # Graphs that have few high (>2) degree vertices on the outer face are not suitable for this algorithm
    if sum(1 if x[1] > 2 else 0 for x in g.in_degree if x[0] in outer_face) <= 3:  # 3 is arbitrary
        # Doe the same thing as if the outer_face is a triangle-- It will get some functional ordering of faces
        return step_one_face(edge, g, outer_face, positions, geom_dict, bottom)

    bottom_verts = list([n for n in bottom_verts if len(list([e for e in bottom if n in e])) > 2])
    # all_paths = dict(nx.all_pairs_shortest_path(g_prime))
    d = {(len(outer_face)-1, 0): (1, 0)}  # d counts do keep track of penalties
    if outer_face[0] not in g_prime and outer_face[-1] not in g_prime:  # algorithm just doesn't have anywhere to start
        for i in range(len(outer_face)):
            if outer_face[i] in g_prime:
                outer_face = list(outer_face[(i+j)%len(outer_face)] for j in range(len(outer_face)))
                edge = (outer_face[-1], outer_face[0])
                break
    # Make the traversal order:
    for i in range(2, len(outer_face)):  # i=l-u+v
        for v in range(i):
            u = len(outer_face) - i + v
            # Keep track of penalties for leaving bottom
            u_penalty = 1 if u != len(outer_face)-1 and (outer_face[u+1], outer_face[u]) not in bottom else 0
            v_penalty = 1 if v > 0 and (outer_face[v], outer_face[v-1]) not in bottom else 0
            # Only allow paths that don't go back and forth from untraversed outer_face, this causes discontinuities
            bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
            # The last safe node in each direction must be removed as well
            g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
            # some bad edges between two good nodes can still survive?
            bad_edges = [e for e in g_prime.edges if (e in outer_face_edges or e in rev_outer_face_edges) and e not in bottom]
            if any(e in g2.edges for e in bad_edges):
                raise BaseException("I don't think this should happen? But then whats the point?")
            if outer_face[u] not in g_prime or outer_face[v] not in g_prime or not nx.has_path(g2, outer_face[u], outer_face[v]):
                if outer_face[v] not in g_prime or g_prime.degree(outer_face[v]) == 0:  # move the 0 degree vertex
                    d[(u, v)] = (d[(u, v - 1)][0] + v_penalty, 0) if v > 0 else (np.infty, 1)
                elif outer_face[u] not in g_prime or g_prime.degree(outer_face[u]) == 0:
                    d[(u, v)] = (d[(u+1, v)][0] + u_penalty, 1) if u < len(outer_face)-1 else (np.infty, 0)
                else:  # can happen, just move one of them
                    if u + 1 == len(outer_face):
                        d[(u, v)] = (d[(u, v - 1)][0] + v_penalty, 0)
                    elif v - 1 < 0:
                        d[(u, v)] = (d[(u + 1, v)][0] + u_penalty, 1)
                    else:
                        ind = np.argmin([d[(u, v-1)][0] + v_penalty, d[(u+1, v)][0] + u_penalty])
                        d[(u, v)] = ([d[(u, v-1)][0] + v_penalty, d[(u+1, v)][0] + u_penalty][ind], ind)
                continue
            if u+1 == len(outer_face):
                d[(u, v)] = (max(d[(u, v-1)][0] + v_penalty, len(nx.shortest_path(g2, source=outer_face[u], target=outer_face[v]))), 0)
            elif v-1 < 0:
                d[(u, v)] = (max(d[(u+1, v)][0] + u_penalty, len(nx.shortest_path(g2, source=outer_face[u], target=outer_face[v]))), 1)
            else:
                ind = np.argmin([d[(u, v-1)][0] + v_penalty, d[(u+1, v)][0] + u_penalty])
                d[(u, v)] = (max([d[(u, v-1)][0] + v_penalty, d[(u+1, v)][0] + u_penalty][ind], len(nx.shortest_path(g2, source=outer_face[u], target=outer_face[v]))), ind)
    # Use the traceback to get the cut order
    if exit_edge is not None:
        end_u, end_v = outer_face.index(exit_edge[0]), outer_face.index(exit_edge[1])
        if end_v > end_u:
            end_u, end_v = end_v, end_u
    else:
        end_u, end_v = min(d.keys(), key=lambda x: d[x][0] if x[0] - x[1] == 1 else np.infty)
    order = []  # order encodes when to increase u or v-- if order is 0, decrease u, if 1, increase v
    u, v = end_u, end_v
    pathlist = []
    for i in range(1, len(outer_face)-1):
        order = [d[(u,v)][1]] + order
        u, v = (u+1, v) if order[0] else (u, v-1)  # opposite
        bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
        g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
        pathlist = [nx.shortest_path(g2, source=outer_face[u], target=outer_face[v])] + pathlist if outer_face[u] in g_prime and outer_face[v] in g_prime and nx.has_path(g2, outer_face[u], outer_face[v]) else pathlist
    u = len(outer_face)-1
    v = 0
    prev_u = len(outer_face)-1
    prev_v = 0
    traversal = []
    prev_p = list(edge)
    iter = 0
    # while g.size() > 0:
    while iter < len(order):
        change_u = order[iter]
        iter += 1
        if 1 <= g.size()/2 - g.number_of_nodes() + 2 <= 2:  # Only one face (2 counting the outside face)
            # inner_face = max([min([g.traverse_face(*e), g.traverse_face(e[1], e[0])], key=lambda x: len(x)) for e in g.edges()])
            while min(g.out_degree, key=lambda x: x[1])[1] < 2:
                g.remove_node(min(g.out_degree, key=lambda x: x[1])[0])
                if g.size() == 0:  # g was just a path
                    return traversal
            return traversal + [list(g.nodes)]
        elif g.size()/2 - g.number_of_nodes() + 2 == 3:  # Manually handle when there are only two faces (3 w/outer) as well
            faces = []
            for e in g.edges:
                if not any(same_face(g.traverse_face(*e), x) for x in faces):
                    faces.append(g.traverse_face(*e))
            if len(faces) != 3:
                raise BaseException("Did not find 3 faces even though g has 3 faces.")
            # Already have an outer_face, find what is left of it
            # for v in g.nodes:
            #     if v in outer_face:
            #         ind = outer_face.index(v)
            #         if (v, outer_face[(ind+1) % len(outer_face)]) in g.edges:
            #             outer_face = g.traverse_face(v, outer_face[(ind+1) % len(outer_face)])
            #             break
            outer_face = min([g.traverse_face(*e) for e in g.edges],
                             key=lambda x: len(x) if all(vert_in_face(g.nodes, x, positions, geom_dict)) else np.infty)
            for f in faces:
                if not same_face(f, outer_face):
                    traversal += [f]  # Might create a minor problem with order but who cares
            return traversal

        g_prime = copy.deepcopy(g)
        g_prime = remove_planar_edges(g_prime, [x for x in outer_face_edges if x in g.edges])
        if change_u:
            u -= 1
            bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
            g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
            while outer_face[u] not in g_prime or not nx.has_path(g2, outer_face[u], outer_face[v]):
                u -= 1
                bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
                g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
                iter += 1
        else:
            v += 1
            bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
            g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
            while outer_face[v] not in g_prime or not nx.has_path(g2, outer_face[u], outer_face[v]):
                v += 1
                bad_nodes = [n for n in g_prime.nodes if n in outer_face and (n not in bottom_verts and n != outer_face[u] and n != outer_face[v])]
                g2 = make_subgraph(g_prime, list([n for n in g_prime.nodes if n not in bad_nodes])) if len(bad_nodes) > 0 else g_prime
                iter += 1
        # Create new bottom:
        if change_u:
            new_bottom = list([x for x in outer_face[u:prev_u] if any([x in b for b in bottom])])
            if len(new_bottom) == 0 or outer_face[prev_u-1] in new_bottom:
                new_bottom += prev_p if outer_face[prev_u] == prev_p[0] else list(reversed(prev_p))
            else:  # don't know which direction to add but bottom has to be continuous so we have mad a whole loop
                # Add to other end
                new_bottom = prev_p + new_bottom if new_bottom[0] in g[prev_p[0]] else list(reversed(prev_p)) + new_bottom
        else:
            new_bottom = list([x for x in outer_face[prev_v + 1: v + 1] if any([x in b for b in bottom])])
            new_bottom = prev_p + new_bottom if outer_face[prev_v] == prev_p[-1] else list(reversed(prev_p)) + new_bottom
        new_bottom = {(new_bottom[i], new_bottom[i+1]) for i in range(len(new_bottom)-1)}
        new_bottom.update({(b[1], b[0]) for b in new_bottom})
        # bottom.update(new_bottom)
        if u == v:
            # we have nothing left above, induct on below
            # Uses previous up_outer if it exists, which has to have been unique'd
            e = None
            for b in new_bottom:
                if b in g.edges:
                    e = b
                    break
            return traversal + step_one_face(e, g, up_outer, positions, geom_dict, new_bottom) if "up_outer" in locals()\
                else traversal + step_one_face(e, g, outer_face, positions, geom_dict, new_bottom)
        p = nx.shortest_path(g2, source=outer_face[u], target=outer_face[v])  # still correct g2
        if prev_p == p:  # We have made no progress! choose the other edge
            raise BaseException("Something is terribly wrong and no progress can be made.")
        # Create down_outer:
        up_outer = outer_face[v+1: u] + p
        # down_outer = prev_p + list(reversed(p)) if prev_p[0] != p[0] else list(reversed(p)) + prev_p
        down_outer = None
        down_outer = list(reversed(outer_face[u:prev_u])) + p + list(reversed(outer_face[prev_v:v]))
        if prev_p[0] == outer_face[prev_u]:
            down_outer += list(reversed(prev_p))
        else:
            down_outer += prev_p
        up = set()
        down = set()
        for vert in g.nodes:
            if vert_in_face(vert, up_outer, positions, geom_dict):
                up.add(vert)
            if vert_in_face(vert, down_outer, positions, geom_dict):
                down.add(vert)
            if vert not in up and vert not in down:
                raise BaseException("Vert {0} is neither below nor above this intermediate path".format(vert))
        to_rem = []
        for i in range(len(down_outer)):
            if down_outer[i] == down_outer[(i-1) % len(down_outer)]:
                to_rem.append(i)
        for i in reversed(to_rem):
            down_outer.pop(i)
        subproblem = make_subgraph(g, list(down) + p)
        for j in range(len(down_outer)):  # check that down_outer doesn't have any non-directly neighboring edge
            if down_outer.count(down_outer[j]) > 1:
                continue  # these will be handled later
            for neighb in subproblem[down_outer[j]]:
                if neighb in down_outer:
                    if neighb == down_outer[(j+1) % len(down_outer)]:
                        continue
                    if neighb == down_outer[(j-1) % len(down_outer)]:
                        continue
                    # Check that if this face is simple
                    danger_face = down_outer[j:down_outer.index(neighb)+1] if j < down_outer.index(neighb)+1 else down_outer[down_outer.index(neighb):j+1]
                    # Add case for when danger face is in the cyclic part
                    if any([vert_in_face(x, danger_face, positions, geom_dict) for x in g_copy.nodes if x not in subproblem.nodes]):
                        # if (neighb, p[j]) in outer_face_edges or (neighb, p[j]) in rev_outer_face_edges:
                        #     continue  # don't remove the outer face edges
                        subproblem = remove_planar_edges(subproblem, [(neighb, down_outer[j])])
                        new_bottom.remove((neighb, down_outer[j])) if (neighb, down_outer[j]) in new_bottom else ""
                        new_bottom.remove((down_outer[j], neighb)) if (down_outer[j], neighb) in new_bottom else ""
                    # Also check if any edges go outside of down, we don't need those
                    elif not vert_in_face((positions[neighb] + positions[down_outer[j]])/2, down_outer, positions, geom_dict, use_graph=False):
                        subproblem = remove_planar_edges(subproblem, [(neighb, down_outer[j])])
                        new_bottom.remove((neighb, down_outer[j])) if (neighb, down_outer[j]) in new_bottom else ""
                        new_bottom.remove((down_outer[j], neighb)) if (down_outer[j], neighb) in new_bottom else ""
        new_graph = make_subgraph(g, list(up) + p)
        if len(down) == len(p):  # We found no additional nodes below p
            # Find the additional edges to remove
            for j in range(len(p)):
                for neighb in new_graph[p[j]]:
                    if neighb in p:
                        if j < len(p)-1 and neighb == p[j+1]:
                            continue
                        if j > 0 and neighb == p[j-1]:
                            continue
                        # if (neighb, p[j]) in outer_face_edges or (neighb, p[j]) in rev_outer_face_edges:
                        #     continue  # don't remove the outer face edges
                        # Remove this edge from new_graph
                        if not vert_in_face((positions[neighb] + positions[p[j]]) / 2, up_outer, positions, geom_dict, use_graph=False):
                            # remove edges that are not contained in up_outer
                            new_graph = remove_planar_edges(new_graph, [(neighb, p[j])])
        # down_outer = list(dict.fromkeys(down_outer))  # unique values with order
        e = None
        for b in new_bottom:
            if b in subproblem.edges:
                e = b
                break
        if nx.has_bridges(subproblem.to_undirected(reciprocal=True)):
            # Simply remove these edges and induct on each nontrivial component
            bridges = list(nx.bridges(subproblem.to_undirected(reciprocal=True)))
            subproblem = remove_planar_edges(subproblem, bridges)
            new_graphs = nx.connected_components(subproblem)
            for coco in new_graphs:
                if len(coco) > 1:
                    g2 = make_subgraph(subproblem, coco)
                    new_edge = [e for e in g2.edges if e in new_bottom][0]
                    traversal += order_faces(g2, new_edge, positions, geom_dict, new_bottom.intersection(set(g2.edges)))
        else:
            traversal += order_faces(subproblem, e, positions, geom_dict, new_bottom)
        g = copy.deepcopy(new_graph)
        prev_p = copy.deepcopy(p)
        prev_u = u
        prev_v = v
    new_bottom = {(p[i], p[i + 1]) for i in range(len(p) - 1)}
    new_bottom.update({(p[1], p[0]) for p in new_bottom})
    traversal += order_faces(g, (p[0], p[1]), positions, geom_dict, new_bottom)  # need to do one last run
    return traversal


# The above algorithm does not work well for certain degenerate cases, this manually adds edge's face to the traversal
def step_one_face(edge, g, outer_face, positions, geom_dict, bottom):
    outer_face_edges = [(outer_face[i], outer_face[(i + 1) % len(outer_face)]) for i in range(len(outer_face))]
    rev_outer_face_edges = [(outer_face[(i + 1) % len(outer_face)], outer_face[i]) for i in range(len(outer_face))]
    f1 = g.traverse_face(*edge)
    f2 = g.traverse_face(edge[1], edge[0])
    g2 = remove_planar_edges(g, [edge])
    new_face = f2 if same_face(f1, outer_face) else f1
    new_face_edges = {(new_face[i], new_face[(i+1)%len(new_face)]) for i in range(len(new_face))}
    new_face_edges.update({(b[1], b[0]) for b in new_face_edges})
    old_edges = new_face_edges.intersection(bottom)
    new_bottom = bottom.difference(old_edges).union(new_face_edges.difference(old_edges))
    # old_slice = list([b for b in bottom if b in new_face])
    # inds = (bottom.index(old_slice[0]), bottom.index(old_slice[-1]))
    # f_inds = (new_face.index(old_slice[0]), new_face.index(old_slice[-1]))
    # new_slice = new_face[f_inds[0]:f_inds[1]+1] if f_inds[0] < f_inds[1] else new_face[f_inds[1]:f_inds[0]+1]
    # if f_inds[0] < f_inds[1]:  # either new_slice follows bottom orientation and is inside or doesn't and is outside
    #     if same_face(new_face[f_inds[0]:f_inds[1]+1], old_slice):  # different orientation and outside
    #         new_slice = list(reversed(new_face[f_inds[1]:] + new_face[:f_inds[0] + 1]))
    #     else:  # same orientation and inside
    #         new_slice = new_face[f_inds[0]:f_inds[1]+1]
    # else: # either new_slice follows bottom orientation and is outside or doesn't and is inside
    #     if same_face(new_face[f_inds[1]:f_inds[0]+1], old_slice):  # same orientation and outside
    #         new_slice = new_face[f_inds[0]:] + new_face[:f_inds[1] + 1]
    #     else:  # different orientation and inside
    #         new_slice = list(reversed(new_face[f_inds[1]:f_inds[0]+1]))
    # new_bottom = bottom[:inds[0]] + new_slice + bottom[inds[1] + 1:]
    traversal = []
    if nx.has_bridges(g2.to_undirected(reciprocal=True)):
        # Simply remove these edges and induct on each nontrivial component
        bridges = list(nx.bridges(g2.to_undirected(reciprocal=True)))
        subproblem = remove_planar_edges(g2, bridges)
        new_graphs = nx.connected_components(subproblem)
        for coco in new_graphs:
            if len(coco) > 1:
                g2 = make_subgraph(subproblem, coco)
                new_edge = [e for e in g2.edges if e in new_bottom][0]
                traversal += [new_face] + order_faces(g2, new_edge, positions, geom_dict, new_bottom.intersection(set(g2.edges)))
    else:
        e = None
        for b in new_bottom:
            if b in g2.edges:
                e = b
                break
        traversal += [new_face] + order_faces(g2, e, positions, geom_dict, new_bottom)
    return traversal


# Checks if two lists share all elements
def same_face(f1, f2):
    return all(y in f2 for y in f1) and all(y in f1 for y in f2)


if __name__ == "__main__":
    h = nx.triangular_lattice_graph(3, 3)
    # h = nx.grid_graph((25, 25))
    g = nx.PlanarEmbedding()
    g_data = {x: h.neighbors(x) for x in h.nodes}
    positions = {x: h.nodes[x]['pos'] for x in h.nodes}
    # positions = {x: x for x in h.nodes}


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
    # for x in g.nodes:
    #     g.connect_components(x, x)
    success, counterexample = nx.check_planarity(g, counterexample=True)
    if not success:
        raise Exception("Not planar")
    # h = make_subgraph(g, [(2,2), (0,2), (0,1), (0,0), (1,0)])
    # g2 = nx.PlanarEmbedding()
    # g2.add_half_edge_ccw(0, 1, None)
    # g2.add_half_edge_ccw(0, 2, None)
    # g2.add_half_edge_ccw(0, 3, None)
    # g2.add_half_edge_ccw(2, 1, None)
    # g2.add_half_edge_ccw(2, 3, None)
    # g2.add_half_edge_ccw(2, 0, None)
    # g2.add_half_edge_ccw(1, 0, None)
    # g2.add_half_edge_ccw(1, 2, None)
    # g2.add_half_edge_ccw(3, 0, None)
    # g2.add_half_edge_ccw(3, 2, None)
    nx.draw(h, with_labels=True, pos=positions)
    plt.show()

    print(vert_in_face((1,2), [(1,3),(2,0),(0,2)], positions))
    # success = nx.check_planarity(h)
    # if not success:
    #     raise Exception("Not planar")
    trav = order_faces(h, edge=((0,3), (1,3)))
    # trav = order_faces(g2, (0, 2), positions)
    print(trav)
    for i in range(len(trav)):
        f = trav[i]
        g.add_node(i)
        positions[i] = (sum(v[0] for v in f)/4, sum(v[1] for v in f)/4)
    nx.draw(g, pos=positions, with_labels=True)
    plt.show()
