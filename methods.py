import itertools

import numpy as np
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover

from utils import create_networkx_graph, extract_max_new_degree, choose_random, normalize


def find_mds_iterative(adj_matrix, nb_iters, rnds):
    degrees = adj_matrix.sum(axis=1)
    weights = degrees + rnds

    neigh_weights = adj_matrix * np.transpose([weights])
    max_neighs = neigh_weights.argmax(axis=0)

    for iter in range(nb_iters):
        max_idxs, counts = np.unique(max_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix * np.transpose([weights])
        max_neighs = neigh_weights.argmax(axis=0)

    return np.unique(max_neighs).tolist()


def find_mds_max_degree_count(adj_matrix, nb_iters, rnds):
    out_degrees = adj_matrix.sum(axis=1)
    weights = out_degrees + rnds

    neigh_weights = adj_matrix * np.transpose([weights])
    max_degree_neighs = neigh_weights.argmax(axis=0)
    max_dominate_neighs = np.copy(max_degree_neighs)

    for iter in range(nb_iters):
        max_idxs, counts = np.unique(max_dominate_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix * np.transpose([weights])
        max_dominate_neighs = neigh_weights.argmax(axis=0)

    new_adj_matrix = np.zeros_like(adj_matrix)
    new_adj_matrix[max_degree_neighs, max_dominate_neighs] = 1
    new_adj_matrix[max_dominate_neighs, max_degree_neighs] = 1

    vertex_cover = []

    for i in range(len(new_adj_matrix)):
        if new_adj_matrix[i,i] == 1:
            vertex_cover.append(i)
            new_adj_matrix[i,:] = 0
            new_adj_matrix[:,i] = 0

    new_graph = nx.from_numpy_matrix(new_adj_matrix)
    vertex_cover += list(min_weighted_vertex_cover(new_graph))

    return vertex_cover


def find_mds_max_count_seprate_neigh(adj_matrix, nb_iters, rnds):
    out_degrees = adj_matrix.sum(axis=1)
    weights = out_degrees + rnds

    neigh_weights = adj_matrix * np.transpose([weights])
    max_dominate_neighs = neigh_weights.argmax(axis=0)

    for iter in range(nb_iters):
        max_idxs, counts = np.unique(max_dominate_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix * np.transpose([weights])
        max_dominate_neighs = neigh_weights.argmax(axis=0)

    common_neigh_count = np.matmul(adj_matrix, adj_matrix.T)
    common_neigh_count = common_neigh_count[max_dominate_neighs]

    seprate_neighs = (adj_matrix * np.transpose([out_degrees])).T - common_neigh_count
    seprate_neighs[adj_matrix == 0] = np.iinfo(np.int32).min 

    seprate_neighs = seprate_neighs.argmax(axis=1)

    new_adj_matrix = np.zeros_like(adj_matrix)
    new_adj_matrix[seprate_neighs, max_dominate_neighs] = 1
    new_adj_matrix[max_dominate_neighs, seprate_neighs] = 1


    new_graph = nx.from_numpy_matrix(new_adj_matrix)
    vertex_cover = list(min_weighted_vertex_cover(new_graph))

    return vertex_cover


def find_min_dominating_set(graph, totally=False):
    nodes = set(range(len(graph)))
    for m in range(1, len(graph)):
        for subset in itertools.combinations(nodes, m):
            have_neigh = True
            for i in range(len(graph)):
                if not totally and i in subset:
                    continue
                node_have_neigh = False
                for node in subset:
                    if graph[node][i]:
                        node_have_neigh = True
                        break
                if not node_have_neigh:
                    have_neigh = False
                    break
            if have_neigh:
                return m, list(subset)
    return list(range(len(graph)))


def find_approx_min_dominating_set(graph, totally=False):
    check = np.ones(len(graph))
    graph = np.array(graph)
    ans = []
    while sum(check) > 0:
        ind = graph.dot(check).argmax()
        neighs = graph[ind,:].nonzero()[0]
        check[neighs] = 0
        if not totally:
            check[ind] = 0
        ans.append(ind)
    return ans


def first_approx_dominating_set(graph, rnds=None):
    nodes, _ = extract_max_new_degree(graph, 1, rnds)

    approx_dominant = set()
    for node in nodes:
        approx_dominant.add(node[0])

    return list(approx_dominant)


def second_approx_dominating_set(graph, rnds=None):
    edges, _ = extract_max_new_degree(graph, 2, rnds)
    new_graph = []
    for i in range(len(graph)):
        new_graph.append([])
        for j in range(len(graph)):
            new_graph[i].append(0)

    for i, edge in enumerate(edges):
        new_graph[edge[0]][edge[1]] = 1
        new_graph[edge[1]][edge[0]] = 1

    new_graph = create_networkx_graph(new_graph)
    vertext_cover = list(min_weighted_vertex_cover(new_graph))
    return vertext_cover


def third_approx_dominating_set(graph, repeat=[1], rnds=None):
    neigh_graph, probs = extract_max_new_degree(graph, rnds=rnds)
    vertext_covers = []
    new_probs = normalize(probs)
    
    for r in repeat:
        best_vertext_cover_len = np.Inf
        best_vertext_cover = None
        
        for k in range(r):
            edges1, new_i, new_p = choose_random(np.array(new_probs).T, np.array(neigh_graph).T)
            new_p = np.array(normalize(new_p))
            edges2, _, _ = choose_random(new_p.T, new_i.T)

            new_graph = []
            for i in range(len(graph)):
                new_graph.append([])
                for j in range(len(graph)):
                    new_graph[i].append(0)

            for i in range(len(edges1)):
                new_graph[edges1[i]][edges2[i]] = 1
                new_graph[edges2[i]][edges1[i]] = 1

            new_graph = create_networkx_graph(new_graph)
            vertext_cover = list(min_weighted_vertex_cover(new_graph))

            if len(vertext_cover) < best_vertext_cover_len:
                best_vertext_cover_len = len(vertext_cover)
                best_vertext_cover = vertext_cover

        vertext_covers.append(best_vertext_cover)

    return vertext_covers


def forth_approx_dominating_set(graph, rnds=None):
    dom_len, dom_set = find_approx_min_dominating_set(graph, totally=True)
    dom_sub_graph = []

    for i in range(len(graph)):
        dom_sub_graph.append([])
        for j in range(len(graph)):
            dom_sub_graph[i].append(0)
    
    for v in dom_set:
        for i in range(len(graph)):
            if graph[i][v]:
                dom_sub_graph[i][v] = 1
                dom_sub_graph[v][i] = 1

    dominants, _ = extract_max_new_degree(dom_sub_graph, 1, rnds=rnds)
    max_neighs, _ = extract_max_new_degree(graph, 2, rnds=rnds)

    new_graph = []

    for i in range(len(graph)):
        new_graph.append([])
        for j in range(len(graph)):
            new_graph[i].append(0)

    for i in range(len(graph)):
        neigh1 = dominants[i][0]
        neigh2 = max_neighs[i][0]
        if neigh1 == neigh2:
            neigh2 = max_neighs[i][1]
        new_graph[neigh1][neigh2] = 1
        new_graph[neigh2][neigh1] = 1

    new_graph = create_networkx_graph(new_graph)
    vertext_cover = list(min_weighted_vertex_cover(new_graph))
    return vertext_cover
