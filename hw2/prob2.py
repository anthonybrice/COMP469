# File: prob2.py
# Desc: A Sudoku solver. Input is an 81-character, numeric string with 0 for
#   blanks. This solution relies heavily on Peter Norvig's solution found
#   here: http://norvig.com/sudoku.html
#   I've tried to add documentation and comments that explain how this solution
#   makes use of CSP algorithms.
# Author: Anthony Brice

import sys

def cross(a, b):
    return [c + d for c in a for d in b]

rows = "ABCDEFGHI"
columns = "123456789"
digits = columns
squares = cross(rows, columns)
unitList = ([cross(rows, c) for c in columns]
            + [cross(r, columns) for r in rows]
            + [cross(rs, cs) for rs in ("ABC", "DEF", "GHI")
               for cs in ("123", "456", "789")])

units = dict((s, [u for u in unitList if s in u]) for s in squares)

peers = dict((s, set(sum(units[s],[])) - set([s])) for s in squares)

def parseGrid(grid):
    """Takes a Sudoku string and returns a dictionary representation of the grid.
    Keys are Sudoku grid positions and values are either the fixed values given
    in the string or a list of any possible value if the position is blank. No
    constraints are checked yet.

    """
    return dict((s, digits) for s in squares)

def gridValues(grid):
    """Takes a Sudoku string and returns a dictionary representation."""
    chars = [c for c in grid if c in digits or c in "0."]
    return dict(zip(squares, chars))

def assign(values, s, d):
    """Returns values with all other values except d eliminated from values[s] and
    with the constraint from the assignment propagated.

    """
    otherValues = values[s].replace(d, "")
    for d2 in otherValues:
        eliminate(values, s, d2)
    return values

def eliminate(values, s, d):
    """Returns values with d eliminated from values[s] and with constraints
    propagated.

    """
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d,"")
    if len(values[s]) == 1:
        d2 = values[s]
        for s2 in peers[s]:
            eliminate(values, s2, d2)
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 1:
            assign(values, dplaces[0], d)
    return values

def display(values):
    """Pretty prints a Sudoku puzzle."""
    width = 1 + max(len(values[s]) for s in squares)
    line = "+".join(["-" * (width * 3)] * 3)
    for r in rows:
        print "".join(values[r + c].center(width) + ("|" if c in "36" else "")
                      for c in columns)
        if r in "CF": print line
    print

def solve(grid):
    """Solves a Sudoku puzzle."""
    return search(parseGrid(grid))

def search(values):
    """Returns the solution for a given puzzle."""
    if all(len(values[s]) == 1 for s in squares):
        return values

    # Chooses the square with the minimum-remaining values.
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))
                for d in values[s])

def some(seq):
    for e in seq:
        return e

if __name__ == "__main__":
    sol = solve(sys.argv[1])
    display(sol)

# Local Variables:
# flycheck-python-pycompile-executable: "/usr/bin/python2"
# End:
