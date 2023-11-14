# cspDFS.py - Solving a CSP using depth-first constraint.
# AIFCA Python3 code Version 0.9.3 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2021.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from cspExamples import csp0, test_csp


def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def dfs_solver(constraints, context, var_order):
    """generator for all solutions to csp.
    context is an assignment of values to some of the variables.
    var_order  is  a list of the variables in csp that are not in context.
    """
    to_eval = {c for c in constraints if c.can_evaluate(context)}
    if all(c.holds(context) for c in to_eval):
        if var_order == []:
            yield context
        else:
            rem_cons = [c for c in constraints if c not in to_eval]
            var = var_order[0]
            for val in var.domain:
                yield from dfs_solver(rem_cons, merge_two_dicts(context,{var:val}), var_order[1:])

def dfs_solve_all(csp, var_order=None):
    """depth-first CSP solver to return a list of all solutions to csp.
    """
    if var_order == None:    # use an arbitrary variable order
        var_order = list(csp.variables)
    return list( dfs_solver(csp.constraints, {}, var_order))

def dfs_solve1(csp, var_order=None):
    """depth-first CSP solver to find single solution or None if there are no solutions.
    """
    if var_order == None:    # use an arbitrary variable order
        var_order = list(csp.variables)
    gen = dfs_solver(csp.constraints, {}, var_order)
    try:     # Python generators raise an exception if there are no more elements.
        return next(gen)
    except StopIteration:
        return None

if __name__ == "__main__":
    #test_csp(dfs_solve1)
    test_csp(dfs_solve1, csp=csp0)
    csp0.show()



#Try:
#dfs_solve_all(csp1)
# dfs_solve_all(csp2)
# dfs_solve_all(crossword1)
# dfs_solve_all(crossword1d)  # warning: may take a *very* long time!
