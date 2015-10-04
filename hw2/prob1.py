# File: prob1.py
# Desc: A program that solves cryptarithmetic equalities such as "REASON == IT *
#   IS + THERE". Note that this program makes use of `eval()` and as such should
#   only be used in a safe environment.
# Author: Anthony Brice

import sys
import re
import itertools


def _letters(s):
    """Returns the case-sensitive set of letters in the given string."""
    return set([x for x in s if x.isalnum()])

pat = re.compile(r"\b0")
def cryptarithmetic(equality):
    """Returns a list of dictionaries defining all solutions to the given
    cryptarithmetic Boolean expression."""

    letters = _letters(equality)

    # Permutations does a simple form of backtracking over partial assignments.
    perms = itertools.permutations(range(10), len(letters))

    sols = list()
    for perm in perms:
        expr = equality

        # Substitute the letters in expr with the numbers in perm.
        for l, n in zip(letters, perm):
            expr = expr.replace(l, str(n))

        try:
            # If there are no leading zeroes AND the expression evaluates to
            # True, then we have a solution.
            if pat.search(expr) == None and eval(expr):
                sols.append(dict(zip(letters, perm)))
        except ZeroDivisionError:
            continue

    return sols

if __name__ == "__main__":
    sols = cryptarithmetic(sys.argv[1])
    for sol in sols:
        print sol

# Local Variables:
# flycheck-python-pycompile-executable: "/usr/bin/python2"
# End:

#  LocalWords:  cryptarithmetic
