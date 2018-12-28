import subprocess
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import LineString, Polygon
import itertools
import os
import random
import networkx as nx
from networkx.algorithms.approximation.vertex_cover import min_weighted_vertex_cover
import numpy as np

def get_random_polygon(radius, max_size, rounds):
    if not os.path.isfile("random_poly.out"):
        compile_proc = subprocess.Popen(["g++", "random_polygon.cpp", "-o", "random_poly.out", "-lCGAL", "-lgmp"], stdout=subprocess.PIPE)
        compile_out = compile_proc.communicate()
        if compile_out[0] or compile_out[1]:
            print(compile_out)

    run_proc = subprocess.Popen(["./random_poly.out", str(radius), str(max_size), str(rounds)], stdout=subprocess.PIPE)
    run_out = run_proc.communicate()

    cordinates = run_out[0].decode("utf-8").split("\n")
    cordinates = [x.split(" ")[1:-1] for x in cordinates]
    points = []
    for j in range(rounds):
        points.append([])
        for i in range(0, len(cordinates[j]), 2):
            points[j].append((int(cordinates[j][i]) + radius, int(cordinates[j][i+1]) + radius))

    return points

def draw_polygon(points, radius):
    image = Image.new("RGB", (2*radius, 2*radius))

    draw = ImageDraw.Draw(image)
    draw.polygon((points), fill=(0,0,255,255))

    image.show()

def get_visibility_graph(points):
    # create polygon
    coords = [p for p in points]
    coords.append(points[0])
    poly = Polygon(coords)

    adj = []
    for i in range(len(points)):
        adj.append([])
        for j in range(len(points)):
            if i == j:
                adj[i].append(0)
                continue
            line = LineString([(points[i][0], points[i][1]), (points[j][0], points[j][1])])
            if line.relate(poly)[2] == 'F':
                adj[i].append(1)
            else:
                adj[i].append(0)

    return adj

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


def extract_max_new_degree(graph, k):
    rnds = [random.uniform(0, 1) for _ in range(len(graph))]
    degrees = [sum(graph[i]) for i in range(len(graph))]
    new_degrees = [x + y for x, y in zip(rnds, degrees)]

    max_new_degrees = []
    for point1 in graph:
        neigh_degrees = []
        for i, point2 in enumerate(point1):
            if point2 == 1:
                neigh_degrees.append(new_degrees[i])
            else:
                neigh_degrees.append(0)
        sorted_inds = sorted(range(len(neigh_degrees)), key=lambda k: neigh_degrees[k])
        max_new_degrees.append(sorted_inds[-k:])

    return max_new_degrees

def first_approx_dominating_set(graph):
    nodes = extract_max_new_degree(graph, 1)

    approx_dominant = set()
    for node in nodes:
        approx_dominant.add(node[0])

    return len(approx_dominant), list(approx_dominant)

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

def second_approx_dominating_set(graph):
    edges = extract_max_new_degree(graph, 2)
    new_graph = []
    for i in range(len(graph)):
        new_graph.append([])
        for j in range(len(graph)):
            new_graph[i].append(0)

    for edge in edges:
        new_graph[edge[0]][edge[1]] = 1
        new_graph[edge[1]][edge[0]] = 1

    new_graph = create_networkx_graph(new_graph)
    vertext_cover = list(min_weighted_vertex_cover(new_graph))
    return len(vertext_cover), vertext_cover

def run():
    radius = 200
    max_size = 300
    rounds = 10
    approx1, approx2 = 0.0, 0.0

    point_sets = get_random_polygon(radius, max_size, rounds)

    for points in point_sets:  
        vis_graph = get_visibility_graph(points)
        x1, _ = find_approx_min_dominating_set(vis_graph, totally=True)
        x2, _ = first_approx_dominating_set(vis_graph)
        x3, _ = second_approx_dominating_set(vis_graph)
        approx1 += (float)(x2) / x1
        approx2 += (float)(x3) / x1
        print(x1, x2, x3)

    approx1 /= rounds
    approx2 /= rounds
    print(approx1, approx2)




if __name__ == '__main__':
    run()
