#test

from cspExamples import n_queens
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

num_queens = 8  # The number of queens to place (and thus the size of the board)

def plot_solution(solution) -> None:
    """Given a solution, plot it and save the result to disk."""
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, num_queens))
    ax.set_ylim((0, num_queens))

    count = 0
    for queen in solution:
        #print (queen)
        ax.add_patch(Rectangle((solution[queen], count), 1, 1))
        count += 1
    fig.savefig(''.join([str(a) for a in solution]) + '.png', dpi=150, bbox_inches='tight')
    #plt.close(fig)


def main() -> None:
    solutions = list(dfs_solve_all(n_queens(8)))
    print('In total: %d solutions' % len(solutions))

    count = 1
    for solution in solutions:
        print('Plotting solution %d/%d: ' % (count, len(solutions)) + str(solution))
        plot_solution(solution)
        count += 1


if __name__ == '__main__':
    main()
