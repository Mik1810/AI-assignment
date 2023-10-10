# agentTop.py - Top Layer
# AIFCA Python3 code Version 0.9.3 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation
import time

import matplotlib

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2021.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from agentMiddle import Rob_middle_layer
from agents import Environment
import matplotlib.pyplot as plt
import math


class Rob_top_layer(Environment):
    def __init__(self, middle, timeout=200, locations={'mail': (-5, 10),
                                                       'o103': (50, 10), 'o109': (100, 10), 'storage': (101, 51)}):
        """middle is the middle layer
        timeout is the number of steps the middle layer goes before giving up
        locations is a loc:pos dictionary 
            where loc is a named location, and pos is an (x,y) position.
        """
        self.middle = middle
        self.timeout = timeout  # number of steps before the middle layer should give up
        self.locations = locations

    def do(self, plan):
        """carry out actions.
        actions is of the form {'visit':list_of_locations}
        It visits the locations in turn.
        """

        to_do = plan['visit']
        to_do_length = len(to_do)
        for i in range(to_do_length):
            rob_x, rob_y = self.middle.env.rob_x, self.middle.env.rob_y
            print(f"Rob x: {rob_x}, Rob y: {rob_y}")
            distance_list = [self.euclidean_distance(rob_x, rob_y, *self.locations[pos]) for pos in to_do]
            print("Distance List: ", distance_list)
            next_pos_index = distance_list.index(min(distance_list))
            print("Position", self.locations[to_do[next_pos_index]], "To do: ", to_do, "Locations: ",self.locations)
            arrived = self.middle.do({'go_to': self.locations[to_do[next_pos_index]], 'timeout': self.timeout})
            self.display(1, "Arrived at", to_do[next_pos_index], arrived)
            if arrived:
                to_do.pop(next_pos_index)

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

class Plot_env(object):
    def __init__(self, body, top):
        """sets up the plot
        """
        self.body = body
        plt.ion()
        plt.clf()
        plt.axes().set_aspect('equal')
        for wall in body.env.walls:
            ((x0, y0), (x1, y1)) = wall
            plt.plot([x0, x1], [y0, y1], "-k", linewidth=3)
        for loc in top.locations:
            (x, y) = top.locations[loc]
            plt.plot([x], [y], "k<")
            plt.text(x + 1.0, y + 0.5, loc)  # print the label above and to the right
        plt.plot([body.rob_x], [body.rob_y], "go")
        plt.draw()


    def plot_run(self):
        """plots the history after the agent has finished.
        This is typically only used if body.plotting==False
        """
        xs, ys = zip(*self.body.history)
        plt.plot(xs, ys, "go")
        wxs, wys = zip(*self.body.wall_history)
        plt.plot(wxs, wys, "ro")
        plt.show()


from agentEnv import Rob_body, Rob_env

#env = Rob_env({((20, 0), (30, 20)), ((70, -5), (70, 25))})

"""
    New environment defined by us
    1. Create a new different environment.
"""
env = Rob_env({((20, 0), (20, 20)),
               ((40, 20), (40, 50)),
              ((70, 30), (120, 30)),
               ((70, 0), (70, 15))}) # ,((70, 30), (120, 30))
body = Rob_body(env)
middle = Rob_middle_layer(body)
top = Rob_top_layer(middle)

# try:
pl = Plot_env(body, top)
top.do({'visit': ['o109', 'storage', 'o103', 'mail']})
pl.plot_run()
# You can directly control the middle layer:
# middle.do({'go_to': (30, -10), 'timeout': 200})
# Can you make it crash?

# Robot Trap for which the current controller cannot escape:
#trap_env = Rob_env({((10, -21), (10, 0)), ((10, 10), (10, 31)), ((30, -10), (30, 0)),
#                    ((30, 10), (30, 20)), ((50, -21), (50, 31)), ((10, -21), (50, -21)),
#                    ((10, 0), (30, 0)), ((10, 10), (30, 10)), ((10, 31), (50, 31))})
#trap_body = Rob_body(trap_env, init_pos=(-1, 0, 90))
#trap_middle = Rob_middle_layer(trap_body)
#trap_top = Rob_top_layer(trap_middle, locations={'goal': (71, 0)})

# Robot trap exercise:
#pl = Plot_env(trap_body, trap_top)
#trap_top.do({'visit':['goal']})