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


# Test if vertex v is contained in (not necessarily convex) ordered face using the dictionary positions
def vert_in_face(v, face, positions):
    if any([x == v for x in face]):
        return True
    p = positions[v]
    inside = False
    for i in range(len(face)):
        if (positions[face[i]][1] > p[1]) != (positions[face[(i+1) % len(face)]][1] > p[1]) and p[0] <= (positions[face[i]][0] - positions[face[(i+1) % len(face)]][0]) * (p[1] - positions[face[(i+1) % len(face)]][1]) / (positions[face[i]][1] - positions[face[(i+1)%len(face)]][1]) + positions[face[(i+1)%len(face)]][0]:
            inside = not inside
    return inside


def vert_in_face_mtpltlib(v, face, positions):
    if any([x == v for x in face]):
        return True
    path = mpath.Path(list([positions[x] for x in face]))
    return path.contains_point(positions[v])


# Given a planar graph g, generate a face order starting with edge
# @param g_copy: The input graph g. this will be copied into a new graph that can change, but we still need to access it
# @param edge: The edge to start the traversal with
# @param positions: the position dictionary of every node in g and maybe more
def order_faces(g_copy, edge, positions):
    g = copy.deepcopy(g_copy)  # The graph to be modified in each iteration
    g_prime = copy.deepcopy(g)  # The version of g with outer edges removed
    # outer_face = max([g.traverse_face(*edge), g.traverse_face(edge[1], edge[0])], key=lambda x: len(x))
    outer_face = min([g.traverse_face(*edge), g.traverse_face(edge[1], edge[0])], key=lambda x: len(x) if all([vert_in_face_mtpltlib(y, x, positions) for y in g.nodes]) else np.infty)
    if not all([vert_in_face_mtpltlib(y, outer_face, positions) for y in g.nodes]):
        raise BaseException("Whats up with this face?")
    outer_face = outer_face[1:] + [outer_face[0]]
    outer_face_edges = [(outer_face[i], outer_face[(i + 1) % len(outer_face)]) for i in range(len(outer_face))]
    rev_outer_face_edges = [(outer_face[(i + 1) % len(outer_face)], outer_face[i]) for i in range(len(outer_face))]

    if nx.has_bridges(g.to_undirected(reciprocal=True)):
        # Simply remove these edges and induct on each nontrivial component
        bridges = list(nx.bridges(g.to_undirected(reciprocal=True)))
        g = remove_planar_edges(g, bridges)
        new_graphs = nx.connected_components(g)
        local_traversal = []
        for coco in new_graphs:
            if len(coco) > 1:
                g2 = make_subgraph(g, coco)
                new_edge = edge if edge in g2.edges else ""
                if new_edge == "":
                    for e in g2.edges:
                        if np.any(e[0] in b or e[1] in b for b in bridges):  # must hold for some edge
                            new_edge = e  # may have a counterexample where we break contiguity!!
                # next_outer = list([e for e in outer_face_edges + rev_outer_face_edges if e in g2.edges])
                # next_outer.remove(new_edge)
                # next_outer.remove((new_edge[1], new_edge[0]))
                # if len(set(next_outer).intersection(set(parent_outer))) == 0:
                #     # Would fail to end on the parent's outer face, jump edges-- any other edge would work
                #     new_edge = outer_face_edges[(outer_face_edges.index(edge) + len(outer_face_edges)/2) % len(outer_face_edges)] if edge in outer_face_edges else rev_outer_face_edges[(rev_outer_face_edges.index(edge) + len(rev_outer_face_edges)/2) % len(rev_outer_face_edges)]
                local_traversal += order_faces(g2, new_edge, positions)
        return local_traversal

    g_prime.remove_edges_from(outer_face_edges)
    g_prime.remove_edges_from(rev_outer_face_edges)
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
        return step_one_face(edge, g, outer_face, positions)
    # Graphs that have few high (>2) degree vertices on the outer face are not suitable for this algorithm
    if sum(1 if x[1] > 2 else 0 for x in g.in_degree if x[0] in outer_face) <= 3:  # 3 is arbitrary
        # Doe the same thing as if the outer_face is a triangle-- It will get some functional ordering of faces
        return step_one_face(edge, g, outer_face, positions)

    all_paths = dict(nx.all_pairs_shortest_path(g_prime))
    u = -1
    v = 0
    prev_u = -1
    prev_v = 0
    traversal = []
    prev_p = list(edge)
    iter = 0
    while g.size() > 0:
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
            for v in g.nodes:
                if v in outer_face:
                    ind = outer_face.index(v)
                    if outer_face[(ind+1) % len(outer_face)] in g.nodes:
                        outer_face = g.traverse_face(v, outer_face[(ind+1) % len(outer_face)])
                        break
            # outer_face = min([g.traverse_face(*edge), g.traverse_face(edge[1], edge[0])],
            #                  key=lambda x: len(x) if all([vert_in_face_mtpltlib(y, x, positions) for y in g.nodes]) else np.infty)
            for f in faces:
                if not same_face(f, outer_face):
                    traversal += [f]
            return traversal
        u_offset = 1 if iter > 1 else 0
        v_offset = 1 if iter > 1 else 0
        # Take care of isolated vertices
        while g_prime.degree(outer_face[v]) == 0 or outer_face[v] not in g.nodes:
            v += 1
            v_offset = 0
        while g_prime.degree(outer_face[u]) == 0 or outer_face[u] not in g.nodes:
            u -= 1
            u_offset = 0
        if u % len(outer_face) <= v % len(outer_face):
            # Finished traversing, induct on whatever is left
            # e = (outer_face[prev_u], outer_face[(prev_u-1) % len(outer_face)]) if prev_u != u else (outer_face[prev_v], outer_face[(prev_v+1) % len(outer_face)])
            e = (prev_p[math.floor(len(prev_p) / 2)], prev_p[math.floor(len(prev_p) / 2) + 1])  # prev_p connects
            if g_copy == g:  # we have not made a single step, just step on some face and induct
                return step_one_face(edge, g, outer_face, positions)
            return traversal + order_faces(g, e, positions)
        while outer_face[v] not in all_paths[outer_face[(u-u_offset) % len(outer_face)]] or (v-(u-u_offset))%len(outer_face) == 0 or outer_face[(u-u_offset) % len(outer_face)] not in g.nodes:
            u_offset += 1
            if u_offset >= len(outer_face):  # no edge going forward from outer_face[v]
                u_offset = 0
                v = (v + 1) % len(outer_face)
                while g_prime.degree(outer_face[v]) == 0:
                    v = (v + 1) % len(outer_face)
                if u % len(outer_face) <= v % len(outer_face):
                    # Finished traversing, induct on whatever is left
                    # e = (outer_face[prev_u], outer_face[(prev_u-1) % len(outer_face)]) if prev_u != u else (outer_face[prev_v], outer_face[(prev_v+1) % len(outer_face)])
                    e = (prev_p[math.floor(len(prev_p)/2)], prev_p[math.floor(len(prev_p)/2) + 1])  # prev_p connects
                    if g_copy == g:  # we have not made a single step, just step on some face and induct
                        return step_one_face(edge, g, outer_face, positions)
                    return traversal + order_faces(g, e, positions)
        while outer_face[(v+v_offset) % len(outer_face)] not in all_paths[outer_face[u]] or (v+v_offset-u)%len(outer_face) == 0 or outer_face[(v+v_offset) % len(outer_face)] not in g.nodes:
            v_offset += 1
            if v_offset >= len(outer_face):  # no edge going forward from outer_face[v]
                v_offset = 0
                u = (u - 1) % len(outer_face)
                while g_prime.degree(outer_face[u]) == 0:
                    u = (u - 1) % len(outer_face)
                if u % len(outer_face) <= v % len(outer_face):
                    # Finished traversing, induct on whatever is left
                    # e = (outer_face[prev_u], outer_face[(prev_u-1) % len(outer_face)]) if prev_u != u else (outer_face[prev_v], outer_face[(prev_v+1) % len(outer_face)])
                    e = (prev_p[math.floor(len(prev_p)/2)], prev_p[math.floor(len(prev_p)/2) + 1])  # prev_p connects
                    if g_copy == g:  # we have not made a single step, just step on some face and induct
                        return step_one_face(edge, g, outer_face, positions)
                    return traversal + order_faces(g, e, positions)
        # p1 = nx.shortest_path(g, source=outer_face[u-1], target=outer_face[v])
        p1 = all_paths[outer_face[(u-u_offset) % len(outer_face)]][outer_face[v]]# if outer_face[v] in all_paths[outer_face[u-1]] else ""
        # p2 = nx.shortest_path(g, source=outer_face[u], target=outer_face[v+1])
        p2 = all_paths[outer_face[u]][outer_face[(v+v_offset) % len(outer_face)]]# if outer_face[v+1] in all_paths[outer_face[u]] else ""
        p = min(p1, p2, key=lambda x: len(x))  # greedy
        up_outer = outer_face[(v+v_offset+1) % len(outer_face): u % len(outer_face)] + p if p == p2 \
            else outer_face[(v+1) % len(outer_face): (u-u_offset) % len(outer_face)] + p
        # down_outer = prev_p + list(reversed(p)) if prev_p[0] != p[0] else list(reversed(p)) + prev_p
        down_outer = None
        # Create down_outer:
        u_end = (u-u_offset) if p == p1 else u
        v_end = (v+v_offset) if p == p2 else v
        if p[0] == outer_face[u_end]:
            down_outer = outer_face[u_end:prev_u] + p + list(reversed(outer_face[prev_v:v_end]))
        else:
            down_outer = outer_face[u_end:prev_u] + list(reversed(p)) + list(reversed(outer_face[prev_v:v_end]))
        if prev_p[0] == outer_face[prev_u]:
            down_outer += list(reversed(prev_p))
        else:
            down_outer += prev_p
        # if prev_p[0] != p[0]:
        #     # Usual case where they do later meet
        #     for off in range(min(len(p)-1, len(prev_p)-1)):
        #         for x in range(min(len(p), len(prev_p)-off)):
        #             if p[x] == prev_p[x+off]:
        #                 # p and prev_p meet at i, i+off
        #                 down_outer = p[:x+1] + list(reversed(prev_p[x+off]))
        #                 break
        #         for x in range(min(len(p)-off, len(prev_p))):
        #             if p[x+off] == prev_p[x]:
        #                 # p and prev_p meet at i+off, i
        #                 down_outer = p[:x+off] + list(reversed(prev_p[:x+1]))
        #                 break
        #         if down_outer is not None:
        #             break
        # elif prev_p[-1] != p[-1]:
        #     # Usual case where they do later meet
        #     for off in range(min(len(p)-1, len(prev_p)-1)):
        #         for x in range(min(len(p), len(prev_p)-off)):
        #             if p[-x-1] == prev_p[-x-off-1]:
        #                 # p and prev_p meet at i, i+off
        #                 down_outer = list(reversed(prev_p[-x-off+1:])) + p[-x-1:]
        #                 break
        #         for x in range(min(len(p)-off, len(prev_p))):
        #             if p[-x-off-1] == prev_p[-x-1]:
        #                 # p and prev_p meet at i+off, i
        #                 down_outer = list(reversed(prev_p[-x:])) + p[-x-off:]
        #                 break
        #         if down_outer is not None:
        #             break
        if down_outer is None:
            raise BaseException("Failed to create down_outer")
        up = set()
        down = set()
        for vert in g.nodes:
            if vert_in_face_mtpltlib(vert, up_outer, positions):
                up.add(vert)
            if vert_in_face_mtpltlib(vert, down_outer, positions):
                down.add(vert)
            if vert not in up and vert not in down:
                raise BaseException("Vert {0} is neither below nor above this intermediate path".format(vert))
        # g.remove_nodes_from(p)
        # comps = list(nx.connected_components(g))
        # up, down = None, None
        # if len(comps) == 1:  # difference is only contained in a face or we are at the end
        #     # Test if we are at the end
        #     f1 = g_copy.traverse_face(p[0], p[1])
        #     f2 = g_copy.traverse_face(p[1], p[0])
        #     if np.all([x in f1 for x in g.nodes]):
        #         # We are at the end
        #         return traversal + [f1]
        #     if np.all([x in f2 for x in g.nodes]):
        #         # We are at the end
        #         return traversal + [f2]
        #     # Check for case when up is none but down is contained
        #     # Find the face:
        #     sym_dif = []
        #     # See if rest of the graph is completely contained between prev_p and p
        #     # How? Use positions? - yes :(
        #     if len(set(p).intersection(set(prev_p))) == 0:
        #         raise BaseException("What?")
        #     if prev_p[0] != p[0]:  # End where we jumped previously
        #         new_edge = (p[0], prev_p[0])
        #         for i in range(min(len(p), len(prev_p))):
        #             if prev_p[i] != p[i]:
        #                 # sym_dif.add(prev_p[i])
        #                 # sym_dif.add(p[i])
        #                 sym_dif = sym_dif[:i] + [p[i], prev_p[i]] + sym_dif[i:]
        #             else:  # Add the first element where they meet
        #                 # sym_dif.add(p[i])
        #                 sym_dif = sym_dif[:i] + [p[i]] + sym_dif[i:]
        #                 # new_edge = (p[i], p[i-1])
        #                 break
        #         # Check that we didn't miss anything due to difference in length:
        #         if len(sym_dif) % 2 == 0:  # ended early
        #             if sym_dif[int(len(sym_dif)/2)-1] in g_copy[sym_dif[int(len(sym_dif)/2)]]:
        #                 pass
        #             elif len(prev_p) < len(p):
        #                 sym_dif = sym_dif[:int(len(sym_dif)/2)-1] + p[p.index(sym_dif[int(len(sym_dif)/2)]):p.index(sym_dif[int(len(sym_dif)/2)-1])] + sym_dif[int(len(sym_dif)/2)+1:]
        #             elif len(p) < len(prev_p):
        #                 sym_dif = sym_dif[:int(len(sym_dif)/2) - 1] + prev_p[prev_p.index(sym_dif[int(len(sym_dif)/2) - 1]):prev_p.index(sym_dif[int(len(sym_dif)/2)])] + sym_dif[int(len(sym_dif)/2) + 1:]
        #         if p[0] not in g_copy[prev_p[0]]:  # skipped some edges, just walk along outer face
        #             last_v = p[0]
        #             while last_v not in g_copy[prev_p[0]]:
        #                 last_v = [x for x in set(g_copy[last_v]).intersection(set(outer_face)) if x not in sym_dif][0]
        #                 # sym_dif.add(last_v)
        #                 sym_dif += [last_v]
        #                 new_edge = (last_v, prev_p[0])
        #     elif prev_p[-1] != p[-1]:  # End where we jumped previously
        #         new_edge = (p[-1], prev_p[-1])
        #         for i in range(min(len(p), len(prev_p))):
        #             if prev_p[-i-1] != p[-i-1]:
        #                 sym_dif = sym_dif[:i] + [p[-i-1], prev_p[-i-1]] + sym_dif[i:]
        #                 # sym_dif.add(prev_p[-i-1])
        #                 # sym_dif.add(p[-i-1])
        #             else:  # Add the first element where they meet
        #                 # sym_dif.add(p[-i-1])
        #                 sym_dif = sym_dif[:i] + [p[-i-1]] + sym_dif[i:]
        #                 # new_edge = (p[-i-1], p[-i])
        #                 break
        #         # Check that we didn't miss anything due to difference in length:
        #         if len(sym_dif) % 2 == 0:  # ended early
        #             if sym_dif[int(len(sym_dif)/2) - 1] in g_copy[sym_dif[int(len(sym_dif)/2)]]:
        #                 pass
        #             elif len(prev_p) < len(p):
        #                 sym_dif = sym_dif[:int(len(sym_dif)/2)-1] + list(reversed(p[p.index(sym_dif[int(len(sym_dif)/2)]):p.index(sym_dif[int(len(sym_dif)/2)-1])])) + sym_dif[int(len(sym_dif)/2)+1:]
        #             elif len(p) < len(prev_p):
        #                 sym_dif = sym_dif[:int(len(sym_dif)/2) - 1] + list(reversed(
        #                     prev_p[prev_p.index(sym_dif[int(len(sym_dif)/2) - 1]):prev_p.index(sym_dif[int(len(sym_dif)/2)])])) + sym_dif[
        #                                                                                                      int(len(sym_dif)/2) + 1:]
        #         if p[-1] not in g_copy[prev_p[-1]]:  # skipped some edges, just walk along outer face
        #             last_v = p[-1]
        #             while last_v not in g_copy[prev_p[-1]]:
        #                 last_v = [x for x in set(g_copy[last_v]).intersection(set(outer_face)) if x not in sym_dif][0]
        #                 # sym_dif.add(last_v)
        #                 sym_dif += [last_v]
        #                 new_edge = (last_v, prev_p[-1])
        #     else:
        #         raise BaseException("This really should never happen.")
        #     # make sym_dif into an ordered face
        #     # for vvert in sym_dif:
        #     #     for uvert in g_copy[vvert]:
        #     #         if uvert in sym_dif:
        #     #             f1 = g_copy.traverse_face(vvert, uvert)
        #     #             if len(set(f1).intersection(sym_dif)) == len(sym_dif):  # found the correct face
        #     #                 sym_dif = f1
        #     #             else:
        #     #                 sym_dif = g_copy.traverse_face(uvert, vvert)
        #     # Check if all the vertices in comp are contained in this face
        #     if all([vert_in_face_mtpltlib(x, sym_dif, positions) for x in comps[0]]):  # Case where up is None
        #         g2 = make_subgraph(g_copy, set(sym_dif).union(set(comps[0])))
        #         return traversal + order_faces(g2, new_edge, positions)
        #     # Resets sym_dif
        #     sym_dif = set(p).symmetric_difference(set(prev_p))  # must have at least 2 verts
        #     if len(sym_dif) < 2:  # may be only one, special case for when graph is embedded i n a triangle
        #         traversal += [p, list(comps[0]) + p]
        #         return traversal
        #         # g = make_subgraph(g_copy, list(comps[0]) + p)
        #         # prev_p = copy.deepcopy(p)
        #         # u, v = ((u - u_offset) % len(outer_face), v) if p == p1 else (u, (v + v_offset) % len(outer_face))
        #         # continue
        #     v1 = None
        #     v2 = None
        #     for lv1 in sym_dif:
        #         for lv2 in sym_dif:
        #             if lv1 in g_copy[lv2]:
        #                 v1, v2 = lv1, lv2
        #     if v1 == None or v2 == None:
        #         raise BaseException("Need to account for extra vertex in sym_dif")
        #     f = g_copy.traverse_face(v1, v2)
        #     for vert in f:
        #         if vert not in p and vert not in prev_p:
        #             f = g_copy.traverse_face(v2, v1)
        #             break
        #     traversal += [f]
        #     g = make_subgraph(g_copy, list(comps[0]) + p)
        #     prev_p = copy.deepcopy(p)
        #     u, v = ((u - u_offset) % len(outer_face), v) if p == p1 else (u, (v + v_offset) % len(outer_face))
        #     continue
        # elif len(comps) > 2:  # We have too many connected components
        #     # Only possible with vertices that are connected only to p, which are surrounded by something in g_copy
        #     # How do we identify which side it is on? Know prev_p for sure, just need to check whether any of the
        #     # isolated vertices are in a face.
        #     while len(comps) > 0:
        #         c = comps.pop(0)
        #         if np.any([x in c for x in prev_p]):
        #             down = c if down is None else down.union(c)
        #         elif up is None:  # If any elements are not connected to p in g_copy, they must be above
        #             for y in c:
        #                 connected = False
        #                 for x in p:
        #                     if x in g_copy[y]:
        #                         connected = True
        #                 if not connected:  # At least one vert in this comp is not connected, so it must be up
        #                     up = c if up is None else up.union(c)
        #                     break
        #             if connected:  # Every vertex in comps[i] was connected
        #                 if down is None:  # both up and down are none, no way to identify
        #                     comps.append(c)
        #                     continue
        #                 done = False
        #                 for y in c:
        #                     for adj in g_copy[y]:
        #                         if done:
        #                             break
        #                         f = g_copy.traverse_face(y, adj)
        #                         if len(set(f).intersection(set(outer_face))) == len(set(f)):
        #                             continue  # can't be outer_face
        #                         for candidate in f:
        #                             if candidate in down:
        #                                 down = down.union(c)
        #                                 done = True
        #                                 break
        #                 if not done:
        #                     up = c
        #         else:
        #             done = False
        #             for y in c:
        #                 for adj in g_copy[y]:
        #                     if done:
        #                         break
        #                     for candidate in g_copy.traverse_face(y, adj):
        #                         if candidate in up:
        #                             up = up.union(c)
        #                             done = True
        #                             break
        #             if not done:
        #                 down = c if down is None else down.union(c)
        # else:
        #     down = comps[0] if np.any([x in comps[0] for x in prev_p]) else comps[1]
        #     up = comps[1] if np.any([x in comps[0] for x in prev_p]) else comps[0]
        # subproblem = g_copy.subgraph(list(down) + p).copy()
        subproblem = make_subgraph(g, list(down) + p)
        for j in range(len(p)):  # check that p[j] doesn't border any non-directly neighboring edge in p
            for neighb in subproblem[p[j]]:
                if neighb in p:
                    if j < len(p)-1 and neighb == p[j+1]:
                        continue
                    if j > 0 and neighb == p[j-1]:
                        continue
                    # Check that if this face is simple
                    danger_face = p[j:p.index(neighb)+1] if j < p.index(neighb)+1 else p[p.index(neighb):j+1]
                    path = mpath.Path(list([positions[x] for x in danger_face]))
                    if any(path.contains_point(positions[x]) for x in g_copy.nodes if x not in subproblem.nodes):
                        subproblem = remove_planar_edges(subproblem, [(neighb, p[j])])
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
                        # Remove this edge from new_graph
                        new_graph = remove_planar_edges(new_graph, [(neighb, p[j])])
        if p == p1:
            e = (outer_face[u - 1], outer_face[u]) if (outer_face[u - 1], outer_face[u]) in subproblem.edges \
                else (outer_face[u], outer_face[u+1])
            if e not in subproblem.edges:
                raise BaseException("Failed to find an initial edge for recursive face ordering procedure.")
        else:
            e = (outer_face[v], outer_face[v+1]) if (outer_face[v], outer_face[v+1]) in subproblem.edges \
                else (outer_face[v-1], outer_face[v])
            if e not in subproblem.edges:
                raise BaseException("Failed to find an initial edge for recursive face ordering procedure.")
        traversal += order_faces(subproblem, e, positions)
        g = copy.deepcopy(new_graph)
        prev_p = copy.deepcopy(p)
        u, v = ((u-u_offset) % len(outer_face), v) if p == p1 else (u, (v+v_offset) % len(outer_face))
        prev_u = u
        prev_v = v
    return traversal


# The above algorithm does not work well for certain degenerate cases, this manually adds edge's face to the traversal
def step_one_face(edge, g, outer_face, positions):
    outer_face_edges = [(outer_face[i], outer_face[(i + 1) % len(outer_face)]) for i in range(len(outer_face))]
    rev_outer_face_edges = [(outer_face[(i + 1) % len(outer_face)], outer_face[i]) for i in range(len(outer_face))]
    f1 = g.traverse_face(*edge)
    f2 = g.traverse_face(edge[1], edge[0])
    g2 = remove_planar_edges(g, [edge])
    # next_outer = list([e for e in outer_face_edges + rev_outer_face_edges if e in g2.edges])
    # next_outer.remove(edge)
    if same_face(f1, outer_face):  # Could be optimized
        for i in range(1, len(outer_face)):  # Loops through edges starting at edge[1]
            if not same_face(g.traverse_face(outer_face[i], outer_face[i-1]), f2):  # take this edge
                # offset = 0
                # if len(set(next_outer).intersection(set(parent_outer))) == 0:
                #     # Would fail to end on the parent's outer face, jump edges-- any other edge would work
                #     offset = len(outer_face_edges) / 2
                if not all([x in outer_face for x in f2]):  # Better to step in then keep going around?
                    in_verts = [x for x in f2 if x not in outer_face]
                    neighb = f2[(f2.index(in_verts[0]) - 1) % len(f2)]
                    if g2.in_degree(neighb) < 2:  # would be a bridge edge contradicting traversal reqs
                        neighb = f2[(f2.index(in_verts[0]) + 1) % len(f2)]
                    return [f2] + order_faces(g2, (neighb, in_verts[0]), positions)
                return [f2] + order_faces(g2, (outer_face[i], outer_face[i-1]), positions)
    else:
        for i in range(1, len(outer_face)):  # Loops through edges starting at edge[1]
            if not same_face(g.traverse_face(outer_face[i], outer_face[i-1]), f1):  # take this edge
                if not all([x in outer_face for x in f2]):  # Better to step in then keep going around?
                    in_verts = [x for x in f1 if x not in outer_face]
                    neighb = f1[(f1.index(in_verts[0]) - 1) % len(f1)]
                    if g2.in_degree(neighb) < 2:  # would be a bridge edge contradicting traversal reqs
                        neighb = f1[(f1.index(in_verts[0]) + 1) % len(f1)]
                    return [f1] + order_faces(g2, (neighb, in_verts[0]), positions)
                return [f1] + order_faces(g2, (outer_face[i], outer_face[i-1]), positions)


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
