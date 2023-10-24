import matplotlib.pyplot as plt


class Plot_env:

    def __init__(self, start=(0, 0), goal=(7, 7)):
        self.start = start
        self.goal = goal
        self.ax, self.fig = None, None
        plt.ion()

    def draw_grid(self):

        self.fig, self.ax = plt.subplots()

        # Make an 8x8 grid
        for i in range(8):
            for j in range(8):
                self.ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor="white", fill=True, edgecolor="black"))

        for i in range(1, 9):
            self.ax.text(i - 0.5, -0.5, str(i), fontsize=12, ha='center', va='center')
            self.ax.text(- 0.5,i - 0.5, str(i), fontsize=12, ha='center', va='center')

        # Set limits for the axes
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 8)

        plt.axis('equal')
        plt.axis('off')
        plt.show()
        plt.pause(2)

    def draw_cell(self, x, y, color):
        # Add a colored cell
        self.ax.add_patch(plt.Rectangle((x - 1, y - 1), 1, 1, facecolor=color, fill=True, edgecolor="black"))
        self.fig.canvas.draw()
        plt.show()
        plt.pause(2)
