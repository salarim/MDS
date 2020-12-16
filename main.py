import numpy as np
import pandas as pd

from graph_loaders import MtxGraphLoader, TxtGraphLoader, DatGraphLoader
from methods import find_mds_iterative, \
                    find_mds_max_degree_count, \
                    find_mds_max_count_seprate_neigh, \
                    find_mds_two_max_count
from utils import add_loops
                        


def main():
    M = 5
    nb_iters = 10

    benchmarks = pd.read_csv('benchmarks.csv')
    for index, row in benchmarks.iterrows():
        try:
            if row['format'] == 'mtx':
                loader = MtxGraphLoader(row['url'])
            elif row['format'] == 'txt':
                loader = TxtGraphLoader(row['url'])
            elif row['format'] == 'dat':
                loader = DatGraphLoader(row['instance'], row['url'], 0)
            else:
                continue
        
            adj_matrix = loader.get_adj_matrix()

            min_sol = None
            for i in range(nb_iters):
                sol = []
                rnds = np.random.rand(adj_matrix.shape[0])

                sol.append(len(find_mds_two_max_count(adj_matrix, M, rnds)))
                sol.append(len(find_mds_two_max_count(adj_matrix, M, rnds, add_loops_in_middle=True)))
                sol.append(len(find_mds_two_max_count(add_loops(adj_matrix), M, rnds)))
                sol.append(len(find_mds_iterative(add_loops(adj_matrix), M, rnds)))
                sol.append(len(find_mds_iterative(adj_matrix, M, rnds)))

                if min_sol is None or sol[4] < min_sol[4]:
                    min_sol = sol

            print('{}\t{}\t{}\t{}\t{}\t{}'.format(row['instance'], *min_sol))
            
            if row['format'] != 'dat' and index != len(benchmarks.index)-1:
                loader.clean()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
