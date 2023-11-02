import matplotlib.pyplot as plt


class Plot:


    def __init__(self, size=(8, 8), start=(0, 0), goal=(8, 8), walls=None):
        if walls is None:
            walls = []
        self.x_size, self.y_size = size
        self.start = start
        self.goal = goal
        self.walls = walls
        self.ax, self.fig = None, None
        plt.ion()


    def draw_grid(self):

        self.fig, self.ax = plt.subplots()

        # Make an 8x8 grid
        for x in range(self.x_size):
            for y in range(self.y_size):
                self.ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="white", fill=True, edgecolor="black"))

        for i in range(1, 9):
            self.ax.text(i - 0.5, -0.5, str(i), fontsize=12, ha='center', va='center')
            self.ax.text(-0.5, i - 0.5, str(i), fontsize=12, ha='center', va='center')

        #Draw walls
        for couple in self.walls:
            x, y = couple
            self.ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, facecolor="black", fill=True, edgecolor="black"))

        #Draw start and goal
        start_x, start_y = self.start
        goal_x, goal_y = self.goal
        start_rect = plt.Rectangle((start_x - 1, start_y - 1), 1, 1, facecolor="green", fill=True, edgecolor="black")
        self.ax.add_patch(start_rect)
        self.ax.add_patch(plt.Rectangle((goal_x - 1, goal_y - 1), 1, 1, facecolor="blue", fill=True, edgecolor="black"))

        #Add labels to start and goal cells
        label_s_x = start_x + start_rect.get_width() / 2.0
        label_s_y = start_y + start_rect.get_height() / 2.0
        label_g_x = goal_x + start_rect.get_width() / 2.0
        label_g_y = goal_y + start_rect.get_height() / 2.0
        self.ax.annotate("s", (label_s_x - 1, label_s_y - 1), color='black', weight='bold', fontsize=10, ha='center', va='center')
        self.ax.annotate("g", (label_g_x - 1, label_g_y - 1), color='black', weight='bold', fontsize=10, ha='center', va='center')

        #Set limits for the axes
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 8)

        plt.axis('equal')
        plt.axis('off')
        #plt.show()
        #plt.pause(20)

    def draw_cell(self, x, y, color):
        # Add a colored cell
        self.ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, facecolor=color, fill=True, edgecolor="black"))
        self.fig.canvas.draw()
        plt.pause(0.5)

    def draw_frontier(self, x, y, color):
        self.ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, facecolor=color, fill=True, edgecolor="black"))
        self.fig.canvas.draw()
