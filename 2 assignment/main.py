from plot import Plot
from network import Graph

size = (8, 8)
start = (4, 3)
goal = (3, 6)
walls = [(3, 3), (7, 3), (2, 4), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]

if __name__ == "__main__":
    env = Plot(size=size, start=start, goal=goal, walls=walls)
    env.draw_grid()

    graph = Graph(size=size, start=start, goal=goal, walls=walls)

    #env.draw_cell(3, 3, "red")
    #env.draw_cell(5, 7, "blue")
