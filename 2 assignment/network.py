import json

from queue import LifoQueue
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def get_Graph(self):
        return self.g

    def __init__(self, size, start, goal, walls):
        self.path = []
        self.stack = LifoQueue()
        self.g = nx.Graph()
        self.x_size, self.y_size = size
        self.walls = walls
        self.start = start
        self.goal = goal
        self.positions = {}
        self.frontier_color = set()

        plt.figure(2, figsize=(7, 6))

        # Draw edge
        for x in range(1, self.x_size + 1):
            for y in range(1, self.y_size + 1):
                self.g.add_node(f"({x}, {y})")
                self.positions[f"({x}, {y})"] = (x, y)

        # print(self.positions)
        with open("position.json", "w") as file:
            json.dump(self.positions, file)

        # Draw edges between the nodes
        for i in range(1, self.y_size + 1):
            self.make_vertical_parent(i, self.y_size)
            self.make_horizontal_parent(i, self.x_size)

        # Remove node where the are walls
        for (x, y) in self.walls:
            self.g.remove_node(f"({x}, {y})")

        # Colouring start node with green
        color_map = []
        for node in self.g:
            if node == str(self.start):
                color_map.append('green')
            elif node == str(self.goal):
                color_map.append('blue')
            else:
                color_map.append('white')

        nx.draw(self.g, pos=self.positions, node_color=color_map, node_size=550, font_size=7, with_labels=True, width=3,
                edgecolors="black")
        plt.show()
        # plt.pause(1)

    def make_vertical_parent(self, x, limit):
        for y in range(1, limit + 1):
            if y == limit:
                return
            self.g.add_edge(f"({x}, {y})", f"({x}, {y + 1})")

    def make_horizontal_parent(self, y, limit):
        for x in range(1, limit + 1):
            if x == limit:
                return
            self.g.add_edge(f"({x}, {y})", f"({x + 1}, {y})")

    def dfs(self):

        visited = set()
        self.path.append(str(self.start))
        self.stack.put(str(self.start))
        print("Stack: ", self.stack.queue)
        print("Explored: ", self.path)

        node = self.stack.get()

        while node != str(self.goal):

            a = list(node)
            current_node_x, current_node_y = int(a[1]), int(a[4])

            neighbors = list(self.g.neighbors(f"({current_node_x}, {current_node_y})"))

            if node not in visited:
                visited.add(node)

            # Aggiungo allo stack i nodi mettendo in senso contrario all'esplorazione
            if f"({current_node_x}, {current_node_y - 1})" in neighbors:
                if f"({current_node_x}, {current_node_y - 1})" not in visited:
                    self.stack.put(f"({current_node_x}, {current_node_y - 1})")
            if f"({current_node_x + 1}, {current_node_y})" in neighbors:
                if f"({current_node_x + 1}, {current_node_y})" not in visited:
                    self.stack.put(f"({current_node_x + 1}, {current_node_y})")
            if f"({current_node_x - 1}, {current_node_y})" in neighbors:
                if f"({current_node_x - 1}, {current_node_y})" not in visited:
                    self.stack.put(f"({current_node_x - 1}, {current_node_y})")
            if f"({current_node_x}, {current_node_y + 1})" in neighbors:
                if f"({current_node_x}, {current_node_y + 1})" not in visited:
                    self.stack.put(f"({current_node_x}, {current_node_y + 1})")

            print("Stack: ", self.stack.queue)

            if all(map(lambda x: x in visited, list(self.g.neighbors(f"({current_node_x}, {current_node_y})")))):
                for node in list(self.g.neighbors(f"({current_node_x}, {current_node_y})")):
                    self.path.append(node)

            node = self.stack.get()
            self.path.append(node)
            print("Explored: ", self.path)
            # time.sleep(1)

    def get_next_node(self):
        for node in self.path:
            yield node, list(self.g.neighbors(node))

    def draw_node(self, x, y, node_color, frontier, frontier_color):
        xy = f"({x}, {y})"
        new_node_color = {}

        self.frontier_color.update(set(frontier))

        for node in self.g.nodes:
            if str(node) == str(self.start):
                new_node_color[str(node)] = "green"
            elif str(node) == str(self.goal):
                new_node_color[str(node)] = "blue"
            elif str(node) == xy:
                new_node_color[xy] = node_color
            elif str(node) in self.frontier_color:
                new_node_color[str(node)] = frontier_color
            else:
                new_node_color[str(node)] = "white"

        pos = self.positions  # Recupera le posizioni dei nodi
        nx.draw(self.g, pos=pos, with_labels=True, node_color=list(new_node_color.values()), node_size=550, font_size=7,
                width=3, edgecolors="black")
        plt.show()
