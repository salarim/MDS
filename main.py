import subprocess
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import LineString, Polygon
import itertools

def get_random_polygon(radius, max_size):
    compile_proc = subprocess.Popen(["g++", "random_polygon.cpp", "-o", "random_poly.out", "-lCGAL", "-lgmp"], stdout=subprocess.PIPE)
    compile_out = compile_proc.communicate()
    if compile_out[0] or compile_out[1]:
        print(compile_out)

    run_proc = subprocess.Popen(["./random_poly.out", str(radius), str(max_size)], stdout=subprocess.PIPE)
    run_out = run_proc.communicate()

    cordinates = run_out[0].decode("utf-8").split(" ")[1:-1]
    points = []
    for i in range(0, len(cordinates), 2):
        points.append((int(cordinates[i]) + radius, int(cordinates[i+1]) + radius))

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

def find_min_dominating_set(graph, totally=False):
    nodes = set(range(len(graph)))
    for m in range(1, len(graph)):
        subsets = set(itertools.combinations(nodes, m))
        for subset in subsets:
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
                return m, subset
    return len(graph)

def run():
    radius = 200
    max_size = 20

    points = get_random_polygon(radius, max_size)
    
    vis_graph = get_visibility_graph(points)

    print(find_min_dominating_set(vis_graph))

    draw_visibility_graph(vis_graph, points, radius)



if __name__ == '__main__':
    run()
