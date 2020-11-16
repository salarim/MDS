import random

from graph_loaders import get_random_polygon, get_visibility_graph
from methods import find_approx_min_dominating_set, \
                    first_approx_dominating_set, \
                    second_approx_dominating_set, \
                    third_approx_dominating_set, \
                    forth_approx_dominating_set


def main():
    radius = 400
    max_size = 30
    rounds = 10
    third_algorithm_repeat = 1
    approxes = [0.0]*(2+third_algorithm_repeat+1)

    point_sets = get_random_polygon(radius, max_size, rounds)

    for points in point_sets:  
        vis_graph = get_visibility_graph(points)
        
        results = []
        rnds = [random.uniform(0, 1) for _ in range(len(vis_graph))]
        x1, _ = find_approx_min_dominating_set(vis_graph, totally=True)
        results.append(x1)
        x2, _ = first_approx_dominating_set(vis_graph, rnds)
        results.append(x2)
        x3, _ = second_approx_dominating_set(vis_graph, rnds)
        results.append(x3)
        x4, _ = third_approx_dominating_set(vis_graph, [len(vis_graph)], rnds)
        for x in x4:
            results.append(x)
        x5, _ = forth_approx_dominating_set(vis_graph, rnds)
        results.append(x5)

        approxes[0] += (float)(x2) / x1
        approxes[1] += (float)(x3) / x1
        for i, x in enumerate(x4):
            approxes[i+2] += (float)(x) / x1
        approxes[2+third_algorithm_repeat] += (float)(x5) / x1 
        print(str(len(points)) + ':')
        print(*results)

    approxes = [a / rounds for a in approxes]
    print(*approxes)


if __name__ == '__main__':
    main()
