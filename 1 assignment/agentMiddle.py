# agentMiddle.py - Middle Layer
# AIFCA Python3 code Version 0.9.3 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2021.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from agents import Environment
from agentEnv import line_segments_intersect
from Distance import euclidean_distance
import math
import matplotlib.pyplot as plt


class Rob_middle_layer(Environment):

    def __init__(self, env):
        self.env = env
        self.percepts = env.initial_percepts()
        self.straight_angle = 11  # angle that is close enough to straight ahead
        self.close_threshold = 2  # distance that is close enough to arrived
        self.close_threshold_squared = self.close_threshold ** 2  # just compute it once
        self.timer = 50  # timer to count the number of ticks between an env to the other
        self.target_pos = None  # Position where it have to go

    def initial_percepts(self):
        return {}

    def do(self, action):
        """action is {'go_to':target_pos,'timeout':timeout}
            target_pos is (x,y) pair
            timeout is the number of steps to try
            returns {'arrived':True} when arrived is true
            or {'arrived':False} if it reached the timeout
        """

        def switch_env():
            # Exchanging the walls
            temp = self.env.env.walls
            self.env.env.walls = self.env.env.other_walls
            self.env.env.other_walls = temp

        def plot_new_env():
            # Print the actual walls
            print("Changed environment", self.env.env.walls)

            # Remove old walls
            for wall in self.env.env.other_walls:
                a, b = wall
                x1, y1 = a
                x2, y2 = b

                print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                plt.plot((x1, x2), (y1, y2), "-w", linewidth=4)

            # Draw the new wall
            for wall in self.env.env.walls:
                a, b = wall
                x1, y1 = a
                x2, y2 = b

                print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                plt.plot((x1, x2), (y1, y2), "-k", linewidth=3)

        if 'timeout' in action:
            remaining = action['timeout']
        else:
            remaining = -1  # will never reach 0
        self.target_pos = action['go_to']
        arrived = self.close_enough(self.target_pos)

        while not arrived and remaining != 0:

            if self.timer == 0:
                switch_env()
                plot_new_env()
                self.timer = 50

            self.timer -= 1
            self.percepts = self.env.do({"steer": self.steer(self.target_pos)})
            remaining -= 1
            arrived = self.close_enough(self.target_pos)
        return {'arrived': arrived}

    def steer(self, target_pos):
        # The bot always steers to the left to avoid obstacles
        if self.percepts['whisker']:
            """self.display(3, 'whisker on', self.percepts)
            for wall in self.env.env.walls:
                bot_x, bot_y = self.percepts['rob_x_pos'], self.percepts['rob_y_pos']
                if line_segments_intersect(((bot_x, bot_y),self.target_pos), wall):
                    # devo controllare se qualche coordinata di un muro è 0
                    if 0 in list(wall[0])+list(wall[1]):
                        return "left"
                    d1 = euclidean_distance(bot_x, bot_y, *wall[0])
                    d2 = euclidean_distance(bot_x, bot_y, *wall[1])
                    print("d1, d2: ", d1, d2)
                    if d2 <= d1:
                        return "left"
                    else:
                        return "right"
                    """
            self.display(3, 'whisker on', self.percepts)
            return "left"
            # Modify this to implement clever steering in front of walls
        else:
            gx, gy = target_pos
            rx, ry = self.percepts['rob_x_pos'], self.percepts['rob_y_pos']
            goal_dir = math.acos((gx - rx) / math.sqrt((gx - rx) * (gx - rx)
                                                       + (gy - ry) * (gy - ry))) * 180 / math.pi
            if ry > gy:
                goal_dir = -goal_dir
            goal_from_rob = (goal_dir - self.percepts['rob_dir'] + 540) % 360 - 180
            assert -180 < goal_from_rob <= 180
            if goal_from_rob > self.straight_angle:
                return "left"
            elif goal_from_rob < -self.straight_angle:
                return "right"
            else:
                return "straight"

    def close_enough(self, target_pos):
        gx, gy = target_pos
        rx, ry = self.percepts['rob_x_pos'], self.percepts['rob_y_pos']
        return (gx - rx) ** 2 + (gy - ry) ** 2 <= self.close_threshold_squared