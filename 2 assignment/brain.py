"""
On the grid shown above, number the nodes expanded (in order) for a depth-first search
from s to g, given that the order of the operators is up, left, right, and down. Assume there is
cycle pruning. What is the first path found?
"""
from searchGeneric import Searcher
from searchProblem import Search_problem_from_explicit_graph as Problem

if __name__ == "__main__":
    searcher1 = Searcher(Problem(None, None, None, None, None, None))  # DFS
    searcher1.search()  # find first path
