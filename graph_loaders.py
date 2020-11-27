import os
import gzip
import shutil
import zipfile
import subprocess
import numpy as np
import urllib.request

from scipy.io import mmread
from scipy.sparse import csr_matrix
# from shapely.geometry import LineString, Polygon


class MtxGraphLoader:

    def __init__(self, url):
        self.url = url
        self.name = '.'.join(url.split('/')[-1].split('.')[:-1])

        if not os.path.exists(self.name):
            self.download()

    def download(self):
        os.mkdir(self.name)

        zipfile_path = os.path.join(self.name, self.name + '.zip')
        urllib.request.urlretrieve(self.url, zipfile_path)

        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(self.name)

        os.remove(zipfile_path)

    def get_adj_matrix(self):
        mtx_file_path = os.path.join(self.name, self.name + '.mtx')
        
        with open(mtx_file_path, 'r+') as f:
            content = f.read()
            if content[1] != '%':
                f.seek(0, 0)
                f.write('%' + content)
        adj_matrix = mmread(mtx_file_path)
        return adj_matrix.tocsr()

    def clean(self):
        shutil.rmtree(self.name)


class TxtGraphLoader:

    def __init__(self, url):
        self.url = url
        self.name = '.'.join(url.split('/')[-1].split('.')[:-2])

        if not os.path.exists(self.name):
            self.download()

    def download(self):
        os.mkdir(self.name)

        zipfile_path = os.path.join(self.name, self.name + '.txt.gz')
        txtfile_path = os.path.join(self.name, self.name + '.txt')
        urllib.request.urlretrieve(self.url, zipfile_path)

        with gzip.open(zipfile_path, 'rb') as f_in:
            with open(txtfile_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(zipfile_path)

    def get_adj_matrix(self):
        txtfile_path = os.path.join(self.name, self.name + '.txt')

        ids = set()
        with open(txtfile_path) as f: 
            for line in f:
                if line[0] != '#':
                    line = line.split('\t')
                    v1, v2 = int(line[0]), int(line[1])
                    for v in [v1,v2]:
                        ids.add(v)
        
        ids = list(ids)
        id_to_idx = {id:i for i, id in enumerate(ids)}
        rows = []
        cols = []
        with open(txtfile_path) as f: 
            for line in f:
                if line[0] != '#':
                    line = line.split('\t')
                    v1, v2 = int(line[0]), int(line[1])
                    idx1 = id_to_idx[v1]
                    idx2 = id_to_idx[v2]
                    rows.append(idx1)
                    cols.append(idx2)
        
        adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids)))
        return adj_matrix
    
    def clean(self):
        shutil.rmtree(self.name)


class DatGraphLoader:

    def __init__(self, name, url, index):
        self.url = url
        self.name = name
        self.index = index

        if not os.path.exists('T1'):
            self.download()

    def download(self):
        zipfile_path = 'ProblemInstances.zip'
        urllib.request.urlretrieve(self.url, zipfile_path)

        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall('.')

        os.remove(zipfile_path)
        shutil.rmtree('T2')

    def get_adj_matrix(self):
        v_nb = self.name[self.name.index('V')+1:self.name.index('E')]
        e_nb = self.name[self.name.index('E')+1:]
        file_name = 'Problem.dat_' + v_nb + '_' + e_nb + '_' + str(self.index)
        file_path = os.path.join('T1', self.name, str(self.index), 'Test', file_name)

        start_line = 2 * int(v_nb) + 5
        adj_matrix = np.zeros((int(v_nb), int(v_nb)))
        with open(file_path) as f: 
            for i, line in enumerate(f):
                if i >= start_line:
                    vertex_idx = i - start_line
                    line = [int(x) for x in line.strip().split(' ')]
                    line = np.array(line)
                    adj_matrix[vertex_idx,:] = line

        sparse_adj_matrix = csr_matrix(adj_matrix)
        return sparse_adj_matrix

    def clean(self):
        shutil.rmtree('T1')


#def get_random_polygon(radius, max_size, rounds):
#    if not os.path.isfile("random_poly.out"):
#        compile_proc = subprocess.Popen(["g++", "random_polygon.cpp", "-o", "random_poly.out", "-lCGAL", "-lgmp", "-lboost_thread"], stdout=subprocess.PIPE)
#        compile_out = compile_proc.communicate()
#        if compile_out[0] or compile_out[1]:
#            print(compile_out)
#
#    run_proc = subprocess.Popen(["./random_poly.out", str(radius), str(max_size), str(rounds)], stdout=subprocess.PIPE)
#    run_out = run_proc.communicate()
#
#    cordinates = run_out[0].decode("utf-8").split("\n")
#    cordinates = [x.split(" ")[1:-1] for x in cordinates]
#    points = []
#    for j in range(rounds):
#        points.append([])
#        for i in range(0, len(cordinates[j]), 2):
#            points[j].append((int(cordinates[j][i]) + radius, int(cordinates[j][i+1]) + radius))
#
#    return points
#
#
#def get_visibility_graph(points):
#    # create polygon
#    coords = [p for p in points]
#    coords.append(points[0])
#    poly = Polygon(coords)
#
#    adj = []
#    for i in range(len(points)):
#        adj.append([])
#        for j in range(len(points)):
#            if i == j:
#                adj[i].append(0)
#                continue
#            line = LineString([(points[i][0], points[i][1]), (points[j][0], points[j][1])])
#            if line.relate(poly)[2] == 'F':
#                adj[i].append(1)
#            else:
#                adj[i].append(0)
#
#    return adj

