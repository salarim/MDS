import numpy as np

from graph_loaders import get_random_polygon, get_visibility_graph, \
                          MtxGraphLoader, TxtGraphLoader, DatGraphLoader
from methods import find_approx_min_dominating_set, \
                    first_approx_dominating_set, \
                    second_approx_dominating_set, \
                    third_approx_dominating_set, \
                    forth_approx_dominating_set, \
                    find_mds_iterative, \
                    find_mds_max_degree_count, \
                    find_mds_max_count_seprate_neigh
                        


def main():
    loader = DatGraphLoader('V100E1000', 'http://mail.ipb.ac.rs/~rakaj/home/ProblemInstances.zip', 0)
    adj_matrix = loader.get_adj_matrix()
    loader.clean()

    rnds = np.random.rand(adj_matrix.shape[0])

    sol = find_mds_iterative(adj_matrix, 5, rnds)

    print(len(sol))

if __name__ == '__main__':
    main()
