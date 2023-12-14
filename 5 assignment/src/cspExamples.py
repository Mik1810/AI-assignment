# cspExamples.py - Example CSPs
# AIFCA Python3 code Version 0.9.3 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2021.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

"""
Michael: In questo file vengono definiti i problemi da risolvere attraverso i vincoli e le variabili.
Infine, è presente un test cases per verificare se la solzione ottenuta è ammissibile

"""

from .cspProblem import Variable, CSP, Constraint

def meet_at(p1, p2):
    """returns a function of two words that is true 
                 when the words intersect at postions p1, p2.
    The positions are relative to the words; starting at position 0.
    meet_at(p1,p2)(w1,w2) is true if the same letter is at position p1 of word w1 
         and at position p2 of word w2.
    """

    def meets(w1, w2):
        return w1[p1] == w2[p2]

    meets.__name__ = "meet_at(" + str(p1) + ',' + str(p2) + ')'
    return meets


words = {'add', 'age', 'aid', 'aim', 'air', 'are', 'arm', 'art', 'bad', 'bat', 'bee',
         'boa', 'dim', 'ear', 'eel', 'eft', 'lee', 'oaf'}
one_across = Variable('one_across', words, position=(0, 0))
one_down = Variable('one_down', words, position=(0, 0.3))
two_down = Variable('two_down', words, position=(0, 0.6))
three_down = Variable('three_down', words, position=(0, 0.9))
four_across = Variable('four_across', words, position=(0, 1.2))
five_across = Variable('five_across', words, position=(0, 1.5))
crossword1 = CSP("crossword1",
                 {one_across, one_down, two_down, three_down, five_across, four_across},
                 [
                     Constraint([one_across, one_down], meet_at(0, 0), position=(1, 0.1)),
                     Constraint([one_across, two_down], meet_at(1, 0), position=(1, 0.3)),
                     Constraint([one_across, three_down], meet_at(2, 0), position=(1, 0.5)),

                     Constraint([four_across, one_down], meet_at(0, 1), position=(1, 0.7)),
                     Constraint([four_across, two_down], meet_at(1, 1), position=(1, 0.9)),
                     Constraint([four_across, three_down], meet_at(2, 1), position=(1, 1.1)),

                     Constraint([five_across, one_down], meet_at(0, 2), position=(1, 1.3)),
                     Constraint([five_across, two_down], meet_at(1, 2), position=(1, 1.5)),
                     Constraint([five_across, three_down], meet_at(2, 2), position=(1, 1.7))
                 ])


def is_word(*letters, words=words):
    """is true if the letters concatenated form a word in words"""
    return "".join(letters) in words


letters = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
           "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
           "z"}

# pij is the variable representing the letter i from the left and j down (starting from 0)
p00 = Variable('p00', letters, position=(0.1, 0.85))
p10 = Variable('p10', letters, position=(0.3, 0.85))
p20 = Variable('p20', letters, position=(0.5, 0.85))
p01 = Variable('p01', letters, position=(0.1, 0.7))
p21 = Variable('p21', letters, position=(0.5, 0.7))
p02 = Variable('p02', letters, position=(0.1, 0.55))
p12 = Variable('p12', letters, position=(0.3, 0.55))
p22 = Variable('p22', letters, position=(0.5, 0.55))
p32 = Variable('p32', letters, position=(0.7, 0.55))
p03 = Variable('p03', letters, position=(0.1, 0.4))
p23 = Variable('p23', letters, position=(0.5, 0.4))
p24 = Variable('p24', letters, position=(0.5, 0.25))
p34 = Variable('p34', letters, position=(0.7, 0.25))
p44 = Variable('p44', letters, position=(0.9, 0.25))
p25 = Variable('p25', letters, position=(0.5, 0.1))

crossword1d = CSP("crossword1d",
                  {p00, p10, p20,  # first row
                   p01, p21,  # second row
                   p02, p12, p22, p32,  # third row
                   p03, p23,  # fourth row
                   p24, p34, p44,  # fifth row
                   p25  # sixth row
                   },
                  [Constraint([p00, p10, p20], is_word, position=(0.3, 0.95)),  # 1-across
                   Constraint([p00, p01, p02, p03], is_word, position=(0, 0.625)),  # 1-down
                   Constraint([p02, p12, p22, p32], is_word, position=(0.3, 0.625)),  # 3-across
                   Constraint([p20, p21, p22, p23, p24, p25], is_word, position=(0.45, 0.475)),  # 2-down
                   Constraint([p24, p34, p44], is_word, position=(0.7, 0.325))  # 4-across
                   ])


def test_csp(CSP_solver, csp):
    #
    """CSP_solver is a solver that takes a csp and returns a solution
    csp is a constraint satisfaction problem
    solutions is the list of all solutions to csp
    This tests whether the solution returned by CSP_solver is a solution.
    """

    solutions_c1 = [{five_across: 'art', two_down: 'ear', one_down: 'boa', four_across: 'oaf', three_down: 'eft', one_across: 'bee'},
                    {one_across: 'boa', one_down: 'bee', two_down: 'oaf', three_down: 'art', four_across: 'ear', five_across: 'eft'}]
    print("Type of CSP_solver:", type(CSP_solver))
    sol0 = CSP_solver(csp)

    with open("terminal.txt", "a") as file:
        file.write(f"CSP: {csp.title} \n")
        file.write(f"Solution found: {sol0} \n")
    file.close()

    print(sol0)
    print("CSP: ", csp.title)
    print("Solution found:", sol0)
    assert sol0 in solutions_c1, "Solution not correct for "+str(csp)
    print("Passed unit test")

# csp0.show()
# csp4.show()
# crossword1.show()
