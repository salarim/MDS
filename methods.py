import itertools

import numpy as np
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover

from utils import create_networkx_graph, extract_max_new_degree, choose_random, normalize


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
    return len(graph), list(range(len(graph)))


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
    return len(ans), ans


def first_approx_dominating_set(graph, rnds=None):
    nodes, _ = extract_max_new_degree(graph, 1, rnds)

    approx_dominant = set()
    for node in nodes:
        approx_dominant.add(node[0])

    return len(approx_dominant), list(approx_dominant)


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
    return len(vertext_cover), vertext_cover


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

    return [len(v) for v in vertext_covers], vertext_covers


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
    return len(vertext_cover), vertext_cover
