import random
import itertools

import numpy as np
import networkx as nx
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


def draw_polygon(points, radius):
    image = Image.new("RGB", (2*radius, 2*radius))

    draw = ImageDraw.Draw(image)
    draw.polygon((points), fill=(0,0,255,255))

    image.show()


def draw_visibility_graph(vis_graph, points, radius):
    image = Image.new("RGB", (2*radius, 2*radius))

    draw = ImageDraw.Draw(image)
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if vis_graph[i][j]:
                draw.line([points[i], points[j]], fill=(255,0,0,255))
    
    for i, point in enumerate(points):
        draw.text((point[0],point[1]),str(i),fill=(255,255,255,255))

    image.show()


def create_networkx_graph(graph):
    graph = np.matrix(graph)
    return nx.from_numpy_matrix(graph)


def extract_max_new_degree(graph, k=None, rnds=None):
    if not rnds:
        rnds = [random.uniform(0, 1) for _ in range(len(graph))]
    degrees = [sum(graph[i]) for i in range(len(graph))]
    new_degrees = [x + y for x, y in zip(rnds, degrees)]

    max_new_degrees = []
    max_neigh_degrees = []
    for point1 in graph:
        neigh_degrees = []
        for i, point2 in enumerate(point1):
            if point2 == 1:
                neigh_degrees.append(new_degrees[i])
            else:
                neigh_degrees.append(0)
        sorted_inds = sorted(range(len(neigh_degrees)), key=lambda k: neigh_degrees[k])
        if k is None:
            max_new_degrees.append(sorted_inds)
            max_neigh_degrees.append(sorted(neigh_degrees))
        else:
            max_new_degrees.append(sorted_inds[-k:])
            max_neigh_degrees.append(sorted(neigh_degrees)[-k:])

    return max_new_degrees, max_neigh_degrees


def find_min_vertex_cover(graph):
    edges = []
    nodes = set()
    for i in range(len(graph)):
        for j in range(i+1,len(graph)):
            if graph[i][j] == 1:
                edges.append((i,j))
                nodes.add(i)
                nodes.add(j)
    
    for m in range(1, len(nodes)):
        for subset in itertools.combinations(nodes, m):
            cover = True
            for edge in edges:
                cover2 = False
                for node in subset:
                    if edge[0] == node or edge[1] == node:
                        cover2 = True
                        break
                if not cover2:
                    cover = False
                    break
            if cover:
                return m, list(subset)

    return len(nodes), list(nodes)


def choose_random(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    result = []
    new_probs = []
    new_items = []
    inverse_probs = prob_matrix.T
    inverse_items = items.T
    for i, x in enumerate(k):
        result.append(items[x][i])
        new_probs.append(np.delete(inverse_probs[i], x))
        new_items.append(np.delete(inverse_items[i], x))
    return result, np.array(new_items), np.array(new_probs)


def normalize(probs):
    new_probs = []
    for prob in probs:
        s = sum(prob)
        new_probs.append([(float)(x) / s for x in prob])
    return new_probs


