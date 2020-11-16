import os
import subprocess

from shapely.geometry import LineString, Polygon


def get_random_polygon(radius, max_size, rounds):
    if not os.path.isfile("random_poly.out"):
        compile_proc = subprocess.Popen(["g++", "random_polygon.cpp", "-o", "random_poly.out", "-lCGAL", "-lgmp", "-lboost_thread"], stdout=subprocess.PIPE)
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
