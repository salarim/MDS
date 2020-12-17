import itertools

import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover

from utils import create_networkx_graph, extract_max_new_degree, choose_random, normalize
from utils import add_loops


def find_mds_iterative(adj_matrix, nb_iters, rnds):
    out_degrees = adj_matrix.sum(axis=1).A1
    weights = out_degrees + rnds

    neigh_weights = adj_matrix.multiply(np.transpose([weights]))
    max_neighs = neigh_weights.argmax(axis=0).A1
    
    for iter in range(nb_iters):
        max_idxs, counts = np.unique(max_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix.multiply(np.transpose([weights]))
        max_neighs = neigh_weights.argmax(axis=0).A1
    
    in_degrees = adj_matrix.sum(axis=0).A1
    return np.unique(max_neighs).tolist() + np.where(in_degrees == 0)[0].tolist()


def find_mds_max_degree_count(adj_matrix, nb_iters, rnds, exact=False):
    out_degrees = adj_matrix.sum(axis=1).A1
    weights = out_degrees + rnds
    
    neigh_weights = adj_matrix.multiply(np.transpose([weights]))
    max_degree_neighs = neigh_weights.argmax(axis=0).A1

    neigh_weights = neigh_weights.tocsr()
    neigh_weights[max_degree_neighs, np.arange(adj_matrix.shape[0])] = -1
    second_max_degree_neighs = neigh_weights.argmax(axis=0).A1

    max_dominate_neighs = np.copy(max_degree_neighs)
    
    for iter in range(nb_iters+1):
        max_idxs, counts = np.unique(max_dominate_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix.multiply(np.transpose([weights]))
        max_dominate_neighs = neigh_weights.argmax(axis=0).A1

    for i in range(adj_matrix.shape[0]):
        if  max_degree_neighs[i] == max_dominate_neighs[i]:
            max_degree_neighs[i] = second_max_degree_neighs[i]
    
    rows = np.append(max_degree_neighs, max_dominate_neighs)
    cols = np.append(max_dominate_neighs, max_degree_neighs)
    rows, cols = zip(*set(zip(rows, cols)))

    new_adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=adj_matrix.get_shape())

    vertex_cover = new_adj_matrix.diagonal().nonzero()[0].tolist()
    nonzero_in_rows = new_adj_matrix[np.array(vertex_cover),:].nonzero()
    new_adj_matrix[np.array(vertex_cover)[nonzero_in_rows[0]], nonzero_in_rows[1]] = 0
    nonzero_in_columns = new_adj_matrix[:,np.array(vertex_cover)].nonzero()
    new_adj_matrix[nonzero_in_columns[0], np.array(vertex_cover)[nonzero_in_columns[1]]] = 0
    new_adj_matrix.eliminate_zeros()

    if exact:
        vertex_cover += find_min_vertex_cover(new_adj_matrix.A)[1]
    else:
        new_graph = nx.from_scipy_sparse_matrix(new_adj_matrix)
        vertex_cover += list(min_weighted_vertex_cover(new_graph))

    return vertex_cover


def find_mds_max_count_seprate_neigh(adj_matrix, nb_iters, rnds):
    out_degrees = adj_matrix.sum(axis=1).A1
    weights = out_degrees + rnds

    neigh_weights = adj_matrix.multiply(np.transpose([weights]))
    max_dominate_neighs = neigh_weights.argmax(axis=0).A1
    
    for iter in range(nb_iters+1):
        max_idxs, counts = np.unique(max_dominate_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix.multiply(np.transpose([weights]))
        max_dominate_neighs = neigh_weights.argmax(axis=0).A1

    common_neigh_count = adj_matrix.dot(adj_matrix.T)
    common_neigh_count = common_neigh_count[max_dominate_neighs,:]

    seprate_neighs = adj_matrix.multiply(np.transpose([out_degrees])+1).T - common_neigh_count
    seprate_neighs = seprate_neighs.argmax(axis=1).A1

    rows = np.append(seprate_neighs, max_dominate_neighs)
    cols = np.append(max_dominate_neighs, seprate_neighs)
    rows, cols = zip(*set(zip(rows, cols)))

    new_adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=adj_matrix.get_shape())

    vertex_cover = new_adj_matrix.diagonal().nonzero()[0].tolist()
    nonzero_in_rows = new_adj_matrix[np.array(vertex_cover),:].nonzero()
    new_adj_matrix[np.array(vertex_cover)[nonzero_in_rows[0]], nonzero_in_rows[1]] = 0
    nonzero_in_columns = new_adj_matrix[:,np.array(vertex_cover)].nonzero()
    new_adj_matrix[nonzero_in_columns[0], np.array(vertex_cover)[nonzero_in_columns[1]]] = 0
    new_adj_matrix.eliminate_zeros()

    new_graph = nx.from_scipy_sparse_matrix(new_adj_matrix)
    vertex_cover += list(min_weighted_vertex_cover(new_graph))

    return vertex_cover


def find_mds_two_max_count(adj_matrix, nb_iters, rnds, add_loops_in_middle=False, exact=False):
    out_degrees = adj_matrix.sum(axis=1).A1
    weights = out_degrees + rnds

    neigh_weights = adj_matrix.multiply(np.transpose([weights]))
    max_dominate_neighs = neigh_weights.argmax(axis=0).A1
    
    for it in range(nb_iters+1):
        max_idxs, counts = np.unique(max_dominate_neighs, return_counts=True)
        weights = np.zeros_like(weights)
        for i in range(len(max_idxs)):
            weights[max_idxs[i]] = counts[i]
        weights = weights + rnds

        neigh_weights = adj_matrix.multiply(np.transpose([weights]))
        max_dominate_neighs = neigh_weights.argmax(axis=0).A1

        if it == nb_iters-1 and add_loops_in_middle:
            adj_matrix = add_loops(adj_matrix)

    neigh_weights = neigh_weights.tocsr()
    neigh_weights[max_dominate_neighs, np.arange(adj_matrix.shape[0])] = -1
    second_max_dominate_neighs = neigh_weights.argmax(axis=0).A1

    for i in range(adj_matrix.shape[0]):
        if neigh_weights[second_max_dominate_neighs[i],i] < 1.0:
            second_max_dominate_neighs[i] = max_dominate_neighs[i]
    
    rows = np.append(second_max_dominate_neighs, max_dominate_neighs)
    cols = np.append(max_dominate_neighs, second_max_dominate_neighs)
    rows, cols = zip(*set(zip(rows, cols)))

    new_adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=adj_matrix.get_shape())

    vertex_cover = new_adj_matrix.diagonal().nonzero()[0].tolist()
    nonzero_in_rows = new_adj_matrix[np.array(vertex_cover),:].nonzero()
    new_adj_matrix[np.array(vertex_cover)[nonzero_in_rows[0]], nonzero_in_rows[1]] = 0
    nonzero_in_columns = new_adj_matrix[:,np.array(vertex_cover)].nonzero()
    new_adj_matrix[nonzero_in_columns[0], np.array(vertex_cover)[nonzero_in_columns[1]]] = 0
    new_adj_matrix.eliminate_zeros()

    if exact:
        vertex_cover += find_min_vertex_cover(new_adj_matrix.A)[1]
    else:
        new_graph = nx.from_scipy_sparse_matrix(new_adj_matrix)
        vertex_cover += list(min_weighted_vertex_cover(new_graph))
    
    in_degrees = adj_matrix.sum(axis=0).A1
    return vertex_cover +  np.where(in_degrees == 0)[0].tolist()


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
