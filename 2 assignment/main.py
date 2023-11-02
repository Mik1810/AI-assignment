from plot import Plot, pause
from network import Graph

size = (8, 8)
start = (4, 3)
goal = (3, 6)
walls = [(3, 3), (7, 3), (2, 4), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]

if __name__ == "__main__":
    env = Plot(size=size, start=start, goal=goal, walls=walls)
    env.draw_grid()

    graph = Graph(size=size, start=start, goal=goal, walls=walls)
    graph.dfs()

    gen = graph.get_next_node()

    while True:
        try:
            node, neighbors = next(gen)
            x, y = int(node[1]), int(node[4])

            graph.draw_node(x, y, "red", neighbors, "orange")

            for node in neighbors:
                x1, y1 = int(node[1]), int(node[4])
                env.draw_frontier(x1, y1, "orange")

            env.draw_cell(x, y, "red")

        except StopIteration:
            pause()
            break



