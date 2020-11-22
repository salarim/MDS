import numpy as np
import pandas as pd

from graph_loaders import MtxGraphLoader, TxtGraphLoader, DatGraphLoader
from methods import find_mds_iterative, \
                    find_mds_max_degree_count, \
                    find_mds_max_count_seprate_neigh
                        


def main():
    benchmarks = pd.read_csv('benchmarks.csv')
    for index, row in benchmarks.iterrows():
        if row['format'] == 'mtx':
            loader = MtxGraphLoader(row['url'])
        elif row['format'] == 'txt':
            loader = TxtGraphLoader(row['url'])
        elif row['format'] == 'dat':
            loader = DatGraphLoader(row['instance'], row['url'])
        else:
            continue
        
        adj_matrix = loader.get_adj_matrix()

        sol = []
        rnds = np.random.rand(adj_matrix.shape[0])
        try:
            sol.append(len(find_mds_iterative(adj_matrix, 5, rnds)))
            sol.append(len(find_mds_max_degree_count(adj_matrix, 5, rnds)))
            sol.append(len(find_mds_max_count_seprate_neigh(adj_matrix, 5, rnds)))
        except Exception as e:
            print(e)
        
        print(row['instance'], adj_matrix.shape, sol)
        if row['format'] != 'dat' and index != len(benchmarks.index)-1:
            loader.clean()

if __name__ == '__main__':
    main()
