import json

import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def get_Graph(self):
        return self.g
    def __init__(self, size,  start, goal, walls):
        self.g = nx.Graph()
        self.x_size, self.y_size = size
        self.walls = walls
        self.start = start
        self.goal = str(goal)
        self.positions = {}

        plt.figure(2, figsize=(7, 6))

        #Draw edge
        for x in range(1,self.x_size + 1):
            for y in range(1, self.y_size + 1):
                self.g.add_node(f"({x}, {y})")
                self.positions[f"({x}, {y})"] = (x, y)

        print(self.positions)
        with open("position.json", "w") as file:
            json.dump(self.positions, file)

        #Draw edges between the nodes
        for i in range(1, self.y_size + 1):
            self.make_vertical_parent(i, self.y_size)
            self.make_horizontal_parent(i, self.x_size)

        #Remove nodew where the are walls
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


        nx.draw(self.g, pos=self.positions, node_color=color_map, node_size=550, font_size=7, with_labels=True, width=3, edgecolors="black")
        plt.show()
        plt.pause(5)

    def make_vertical_parent(self, x, limit):
        for y in range(1, limit + 1):
            if y == limit:
                return
            self.g.add_edge(f"({x}, {y})", f"({x}, {y+1})")

    def make_horizontal_parent(self, y, limit):
        for x in range(1, limit + 1):
            if x == limit:
                return
            self.g.add_edge(f"({x}, {y})", f"({x+1}, {y})")
    def dfs(self):
        current_node_x, current_node_y = self.start
        neighbors = list(self.g.neighbors(str(f"({current_node_x}, {current_node_ygit }")))
        while not f"({current_node_x}, {current_node_y})" == self.goal:
             if f"({current_node_x}, {current_node_y+1})" in neighbors:
                current_node_y+=1
                yield f"({current_node_x}, {current_node_y})"
             elif f"({current_node_x-1}, {current_node_y})" in neighbors:
                current_node_x-=1
                yield f"({current_node_x}, {current_node_y})"
             elif f"({current_node_x+1}, {current_node_y})" in neighbors:
                current_node_x+=1
                yield f"({current_node_x}, {current_node_y})"
             elif f"({current_node_x}, {current_node_y-1})" in neighbors:
                current_node_y-=1
                yield f"({current_node_x}, {current_node_y})"


        z = list(self.g.neighbors("(1, 1)")).pop()
        x = int(z[1])
        y = int(z[4])
        #print(x, y)


        """
        for i in range(1, 65):
            self.g.add_edge(i, i + 1)
            self.g.add_edge(i, i + 8)

        for i in range(1, self.x_size+1):
            for j in range(1, self.y_size+1):
                self.positions[i + 8 * (j - 1)] = (i, j)

        print(self.positions)

        for i in range(1, 65):
            nx.set_node_attributes(self.g, i, f"({i}, {i + 1})")




        for couple in self.walls:
            x, y = couple

        #
        nx.draw(self.g, pos=self.positions, node_size = 500, font_size=7, with_labels=True)
        plt.show()
        plt.pause(20)
        
# creazione grafico frocio completo 8x8
for i in range(1, 65):
    g.add_edge(i, i + 1)
    g.add_edge(i, i + 8)

# rimozione dei nodi non presenti sulla griglia originale
g.remove_node(23)
g.remove_node(19)
g.remove_node(26)
for x in range(35, 40):
    g.remove_node(x)
# rimozione dei nodi in più
for j in range(65, 73):
    g.remove_node(j)

# rimozione degli archi in più
cont = 9
while (cont <= 58):
    g.remove_edge(cont, cont - 1)
    print("cont: " + str(cont) + "cont-1: " + str(cont - 1))
    cont += 8

node_positions = {}

for i in range(1, 9):
    for j in range(1, 9):
        node_positions[i + 8 * (j - 1)] = (i, j)
for i in range(1, 65):
    nx.set_node_attributes(g, i, ("(" + str(i) + ")"",(" + str(i + 1) + ")"))
# Disegna il grafo utilizzando le posizioni definite
label = node_positions
nx.draw(g, pos=node_positions, with_labels=True)
plt.show()
"""
        """
 plt.savefig("filename.png")

# Example3 (saving figures to files)
# importing networkx
# importing matplotlib.pyplot


g = nx.Graph()

g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(1, 4)
g.add_edge(1, 5)
g.add_edge(5, 6)
g.add_edge(5, 7)
g.add_edge(4, 8)
g.add_edge(3, 8)

# drawing in circular layout
nx.draw_circular(g, with_labels=True)
plt.savefig("filename1.png")

# clearing the current plot
plt.clf()

#drawing in planar layout
nx.draw_planar(g, with_labels=True)
plt.savefig("filename2.png")

# clearing the current plot
plt.clf()

# drawing in random layout
nx.draw_random(g, with_labels=True)
plt.savefig("filename3.png")

# clearing the current plot
plt.clf()

# drawing in spectral layout
nx.draw_spectral(g, with_labels=True)
plt.savefig("filename4.png")

# clearing the current plot
plt.clf()

# drawing in spring layout
nx.draw_spring(g, with_labels=True)
plt.savefig("filename5.png")

# clearing the current plot
plt.clf()

# drawing in shell layout
nx.draw_shell(g, with_labels=True)
plt.savefig("filename6.png")

# clearing the current plot
plt.clf()

# Example4

G = nx.DiGraph()
G.add_edges_from([(1, 1), (1, 7), (2, 1), (2, 2), (2, 3),
                  (2, 6), (3, 5), (4, 3), (5, 4), (5, 8),
                  (5, 9), (6, 4), (7, 2), (7, 6), (8, 7)])

# plt.figure(figsize =(9, 9))
nx.draw_networkx(G)
# nx.draw_networkx(G,with_labels=True)

# getting different graph attributes
print("Total number of nodes: ", int(G.number_of_nodes()))
print("Total number of edges: ", int(G.number_of_edges()))
print("List of all nodes: ", list(G.nodes()))
print("List of all edges: ", list(G.edges()))
print("In-degree for all nodes: ", dict(G.in_degree()))
print("Out degree for all nodes: ", dict(G.out_degree()))

print("List of all nodes we can go to in a single step from node 2: ",
      list(G.successors(2)))

print("List of all nodes from which we can go to node 2 in a single step: ",
      list(G.predecessors(2)))


G = nx.house_graph()
# explicitly set positions
pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}

# Plot nodes with different properties for the "wall" and "roof" nodes
nx.draw_networkx_nodes(
    G, pos, node_size=3000, nodelist=[0, 1, 2, 3], node_color="tab:blue"
)
nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[4], node_color="tab:orange")
nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
# Customize axes
ax = plt.gca()
ax.margins(0.11)
plt.tight_layout()
plt.axis("off")
plt.show()

G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)

# explicitly set positions
pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}
nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

# Example7
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edge('ts', 'mail', weight=6)
G.add_edge('mail', 'ts', weight=6)
G.add_edge('o103', 'ts', weight=8)
G.add_edge('ts', 'o103', weight=8)
G.add_edge('o103', 'b3', weight=4)
G.add_edge('o103', 'o109', weight=12)
G.add_edge('o109', 'o103', weight=12)
G.add_edge('o109', 'o119', weight=16)
G.add_edge('o119', 'o109', weight=16)
G.add_edge('o109', 'o111', weight=4)
G.add_edge('o111', 'o109', weight=4),
G.add_edge('b1', 'c2', weight=3)
G.add_edge('b1', 'b2', weight=6)
G.add_edge('b2', 'b1', weight=6),
G.add_edge('b2', 'b4', weight=3)
G.add_edge('b4', 'b2', weight=3),
G.add_edge('b3', 'b1', weight=4)
G.add_edge('b1', 'b3', weight=4),
G.add_edge('b3', 'b4', weight=7)
G.add_edge('b4', 'b3', weight=7),
G.add_edge('b4', 'o109', weight=7)
G.add_edge('c1', 'c3', weight=8)
G.add_edge('c3', 'c1', weight=8),
G.add_edge('c2', 'c3', weight=6)
G.add_edge('c3', 'c2', weight=6),
G.add_edge('c2', 'c1', weight=4)
G.add_edge('c1', 'c2', weight=4),
G.add_edge('o123', 'o125', weight=4)
G.add_edge('o125', 'o123', weight=4),
G.add_edge('o123', 'r123', weight=4)
G.add_edge('r123', 'o123', weight=4),
G.add_edge('o119', 'o123', weight=9)
G.add_edge('o123', 'o119', weight=9),
G.add_edge('o119', 'storage', weight=7)
G.add_edge('storage', 'o119', weight=7)

import pygraphviz

# pos = nx.random_layout(G,dim=7,seed=7)
# pos=nx.circular_layout(G,scale=14)
# if you choose to use this layout, you have to install pygraphviz
# !apt install libgraphviz-dev
# !pip install pygraphviz
pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
# pos = nx.spring_layout(G, seed=7,scale=10)  # positions for all nodes - seed for reproducibility
nx.draw_networkx_nodes(G, pos, node_size=500)
edgeslist = [(u, v) for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, edgelist=edgeslist, width=2)
# node labels
nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
plt.axis("off")
# plt.figure(G, figsize=(30,30))
plt.show()
# plt.figure(figsize =(9, 9))
# nx.draw_networkx(G)
"""
