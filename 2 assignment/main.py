import time

from plot import Plot_env


if __name__ == "__main__":
    env = Plot_env()
    env.draw_grid()

    env.draw_cell(3, 3, "red")
    env.draw_cell(5, 7, "blue")
