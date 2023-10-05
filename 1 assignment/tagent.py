
import random
from agents import Environment, Agent

class TP_env(Environment):
    prices = [234, 234, 234, 234, 255, 255, 275, 275, 211, 211, 211,
    234, 234, 234, 234, 199, 199, 275, 275, 234, 234, 234, 234, 255,
    255, 260, 260, 265, 265, 265, 265, 270, 270, 255, 255, 260, 260,
    265, 265, 150, 150, 265, 265, 270, 270, 255, 255, 260, 260, 265,
    265, 265, 265, 270, 270, 211, 211, 255, 255, 260, 260, 265, 265,
    260, 265, 270, 270, 205, 255, 255, 260, 260, 265, 265, 265, 265,
    270, 270]
    max_price_addon = 20  # maximum of random value added to get price

    def __init__(self):
        """paper buying agent"""
        self.time=0
        self.stock=20
        self.stock_history = []  # memory of the stock history
        self.price_history = []  # memory of the price history

    def initial_percepts(self):
        """return initial percepts"""
        self.stock_history.append(self.stock)
        price = self.prices[0]+random.randrange(self.max_price_addon)
        self.price_history.append(price)
        return {'price': price,
                'instock': self.stock}

    def do(self, action):
        """does action (buy) and returns percepts (price and instock)"""
        used = pick_from_dist({6:0.1, 5:0.1, 4:0.2, 3:0.3, 2:0.2, 1:0.1})
        bought = action['buy']
        self.stock = self.stock+bought-used
        self.stock_history.append(self.stock)
        self.time += 1
        price = (self.prices[self.time%len(self.prices)] # repeating pattern
                 +random.randrange(self.max_price_addon) # plus randomness
                 +self.time//2)                          # plus inflation
        self.price_history.append(price)
        return {'price': price,
                'instock': self.stock}

def pick_from_dist(item_prob_dist):
    """ returns a value from a distribution.
    item_prob_dist is an item:probability dictionary, where the
        probabilities sum to 1.
    returns an item chosen in proportion to its probability
    """
    ranreal = random.random()
    for (it,prob) in item_prob_dist.items():
        if ranreal < prob:
            return it
        else:
            ranreal -= prob
    raise RuntimeError(str(item_prob_dist)+" is not a probability distribution")

class TP_agent(Agent):
    def __init__(self, env):
        self.env = env
        self.spent = 0
        percepts = env.initial_percepts()
        self.ave = self.last_price = percepts['price']
        self.instock = percepts['instock']
        self.tobuy_history=[]
        
    def go(self, n):
        """go for n time steps
        """
        for i in range(n):
            if self.last_price < 0.9*self.ave and self.instock < 60:
                tobuy = 48
            elif self.instock < 12:
                tobuy = 12
            else:
                tobuy = 0
            self.spent += tobuy*self.last_price
            percepts = env.do({'buy': tobuy})
            self.last_price = percepts['price']
            self.ave = self.ave+(self.last_price-self.ave)*0.05
            self.instock = percepts['instock']
            self.tobuy_history.append(tobuy)

env = TP_env()
ag = TP_agent(env)
ag.go(90)
ag.spent/env.time   ## average spent per time period

import matplotlib.pyplot as plt

class Plot_prices(object):
    """Set up the plot for history of price and number in stock"""
    def __init__(self, ag,env):
        self.ag = ag
        self.env = env
        plt.ion()
        #plt.xlabel("Time")
        #plt.ylabel("Number in stock.                                              Price.")

    def plot_run(self):
        """plot history of price and instock"""
        num = len(env.stock_history)
        print(env.stock_history)
        print(env.price_history)
        print(ag.tobuy_history)
        #plt.subplot(3,1,1)
        fig, axes=plt.subplots(nrows=3, ncols=1)
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.7)
        axes[0].plot(range(num),env.stock_history,label="In stock")
        axes[0].legend(loc="upper left")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Number in stock")
        #plt.subplot(3,1,2)

        axes[1].plot(range(num),env.price_history,color='r',label="Price")
        axes[1].legend(loc="upper left")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Price")

        #plt.subplot(3,1,3)

        axes[2].plot(range(num-1),ag.tobuy_history,c='green',label="Ammount to Buy")
        axes[2].legend(loc="upper left")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Ammount to buy")
        plt.show()
#...................................
pl = Plot_prices(ag,env)
ag.go(5); pl.plot_run()

