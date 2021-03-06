# File: prob3.py
# Desc: A solution to Problem 6.10 in AIMA
# Author: Anthony Brice

from random import random, choice
from copy import deepcopy
from collections import deque
import networkx
import math
import sys
from timeit import Timer
import csv

########
# MAIN #
########

def main(args):
    l = [mapProblem((x+1)*10) for x in range(10)]
    print "Finished generating graphs"

    min4 = [Timer(lambda: minConflicts(G, 4)).timeit(number=1) for G in l]
    print "Finished timing minConflicts"
    back4 = [Timer(lambda: backtrackingSearch(G, 4)).timeit(number=1)
             for G in l[:5]]
    print "Finished timing backtracking"
    fc4 = [Timer(lambda: backtrackingSearch(G, 4, inference=forwardChecking))
           .timeit(number=1) for G in l]
    print "Finished timing forward checking"
    mac4 = [Timer(lambda: backtrackingSearch(G, 4, inference=maintainingArcConsistency))
            .timeit(number=1) for G in l]
    print "Finished timing MAC"

    out = csv.writer(open("data.csv", "w"), delimiter=",")
    out.writerow(min4)
    out.writerow(back4)
    out.writerow(fc4)
    out.writerow(mac4)

####################
# SEARCH FUNCTIONS #
####################

def minConflicts(G, k):
    """Returns a solution using the min-conflicts algorithm to the given dual graph
    where each vertex is valued from range(k) and no neighboring vertices share
    the same value.

    """
    csp = getCsp(G, k)

    return _minConflicts(csp)

def backtrackingSearch(G, k, inference=None):
    """Returns a solution using the backtracking algorithm to the given dual graph
    where each vertex is valued from range(k) and no neighboring vertices share
    the same value.

    """
    csp = getCsp(G, k)

    return _backtrackingSearch(csp, inference)

def _backtrackingSearch(csp, inference):
    """Returns a solution to the given csp using the backtracking algorithm."""
    # Make initial assignment.
    assignment = dict()
    assignment["assignment"] = dict()
    assignment["inferences"] = dict()
    for var in csp["vars"]:
        assignment["assignment"][var] = None
        assignment["inferences"][var] = list(csp["domains"])

    return backtrack(assignment, csp, inference)["assignment"]

def backtrack(assignment, csp, inference):
    """Returns a solution to given csp using the backtracking algorithm starting at
    assignment.

    """
    if isComplete(assignment["assignment"]):
        return assignment

    var = selectUnassignedVariable(assignment, csp)
    for value in orderDomainValues(var, assignment, csp):
        oldInferences = deepcopy(assignment["inferences"])
        if isConsistent(var, value, assignment["assignment"], csp):
            assignment["assignment"][var] = value
            assignment["inferences"][var] = [value]

            inferences = assignment["inferences"]
            if inference is not None:
                inferences = inference(csp, var, value, assignment)

            if isSuccessfulInferences(inferences):
                assignment["inferences"] = inferences

                result = backtrack(assignment, csp, inference)
                if result is not None:
                    return result

        assignment["assignment"][var] = None
        assignment["inferences"] = oldInferences

    return None

def isSuccessfulInferences(inferences):
    """Returns True if inferences contains no inconsistency."""
    success = True
    for k, v in inferences.items():
        if not v:
            success = False
            break

    return success

def forwardChecking(csp, var, value, assignment):
    """Returns a dictionary of inferences such that for each unassigned variable Y
    that is connected to var by a constraint, delete from Y's domain any value
    that is inconsistent with the value chosen for var.

    """
    inferences = assignment["inferences"]
    for neighbor in csp.get("neighbors").get(var):
        if assignment["assignment"].get(neighbor) is None:
            inferences[neighbor] = [x for x in inferences[neighbor]
                                    if x != value]

    return inferences

def maintainingArcConsistency(csp, var, value, assignment):
    """Returns a dictionary of inferences such that all unassigned variables that
    are neighbors of var have constraints propagated in AC-3's usual way.

    """
    inf = assignment["inferences"]
    queue = deque([(v, var) for v in csp["neighbors"][var]
                   if assignment["assignment"][v] is None])

    while queue:
        Xi, Xj = queue.popleft()
        if revise(csp, assignment, Xi, Xj):
            if len(inf[Xi]) == 0:
                return inf
            for Xk in [X for X in csp["neighbors"][Xi] if X != Xj]:
                queue.append((Xk, Xi))

    return inf

def revise(csp, assignment, Xi, Xj):
    """Returns True iff we revise the domain of Xi."""
    revised = False
    inf = assignment["inferences"]

    for x in inf[Xi]:
        if all(not csp["constraints"](Xi, x, Xj, y)
               for y in inf[Xj]):
            inf[Xi].pop(inf[Xi].index(x))
            revised = True

    return revised


def isSuccess(assignment, csp):
    """Returns True if the given assignment satisfies all constraints."""
    success = True
    for k, v in assignment.items():
        if not isConsistent(k, v, assignment, csp):
            success = False
            break

    return success

def isConsistent(var, value, assignment, csp):
    """Returns True if assigning value to var maintains the assignment's
    consistency.

    """
    consistent = True
    for neighbor in csp["neighbors"][var]:
        val2 = assignment[neighbor]
        if (val2 is not None
            and not csp["constraints"](var, value, neighbor, val2)):
            consistent = False
            break

    return consistent

def selectUnassignedVariable(assignment, csp):
    """Returns the next unassigned variable in assignment."""
    # Return the variable with the minimum-remaining values. Note that this only
    # takes effect if constraints are propagated by an inference procedure.
    l = [x for x in csp["vars"] if assignment["assignment"][x] is None]
    return min(l, key=lambda x: len(assignment["inferences"][x]))

def orderDomainValues(var, assignment, csp):
    """Returns an ordering of the domain values."""
    # Return the values in order of least constraining. For each value, we count
    # the number of values it rules out for all var's neighbors.
    l = list()
    for value in assignment["inferences"][var]:
        num = 0
        for neighbor in csp["neighbors"][var]:
            for val2 in assignment["inferences"][neighbor]:
                if not csp["constraints"](var, value, neighbor, val2):
                    num += 1
        l.append((value, num))

    l.sort(key=lambda x: x[1])

    return map(lambda x: x[0], l)

def isComplete(assignment):
    """Returns True if every key in assignment has a value."""
    return all(v is not None for v in assignment.values())


def _minConflicts(csp, maxSteps=1000000):
    """Returns a solution to the given csp using the min-conflicts algorithm."""
    current = dict()
    domains = csp.get("domains")

    # Make initial assignment.
    for var in csp.get("vars"):
        current[var] = domains[0]

    for i in range(maxSteps):
        c = conflictingVariables(csp, current)
        if not c:
            return current

        var = choice(c)
        value = minValue(var, current, csp)
        current[var] = value

    return None

def minValue(var, current, csp):
    """Returns the value that minimizes the number of conflicts var can have. In the
    event that more than one value minimizes, one of those is returned at
    random.

    """
    l = list()
    for v in csp["domains"]:
        num = conflicts(var, v, current, csp)
        l.append((v, num))

    m = [x for x in l if x[1] == min(y[1] for y in l)]

    return choice(m)[0]

def conflicts(var, v, current, csp):
    """Returns the number of conflicts with other variables var has when given
    the value of v.

    """
    num = 0
    for neighbor in csp["neighbors"][var]:
        val2 = current[neighbor]
        if not csp["constraints"](var, v, neighbor, val2):
            num += 1

    return num

def conflictingVariables(csp, current):
    """Returns a list of conflicted variables in the csp with current state.

    """
    conflicts = list()
    for var in csp["vars"]:
        val = current[var]
        if any(not csp["constraints"](var, val, neighbor, current[neighbor])
               for neighbor in csp["neighbors"][var]):
            conflicts.append(var)

    return conflicts

#################
# CSP GENERATOR #
#################

# Definition takes inspiration from here: http://aima.cs.berkeley.edu/python/csp.html
# The implementation is my own.

def getCsp(G, k):
    """Returns a constraint-satisfaction problem of the given dual graph with domain
    range(k) and the constraint that no vertex can share the value of an
    adjacent vertex.

    """
    csp = dict()
    csp["vars"] = G.nodes()
    csp["domains"] = range(k)
    csp["neighbors"] = getCspNeighbors(G)
    csp["constraints"] = lambda A,a,B,b: a != b

    return csp

def getCspNeighbors(G):
    """Returns a dictionary in which the keys are the vertices of G and the values
    are the neighbors of that key.

    """
    d = dict()
    for node in G.nodes():
        l = list()
        for edge in G.edges():
            if node == edge[0]:
                l.append(edge[1])
            elif node == edge[1]:
                l.append(edge[0])
        d[node] = l

    return d

###################
# GRAPH GENERATOR #
###################

def scatter(n):
    """Returns a graph with n nodes of random location on the unit square."""
    G = networkx.Graph()
    while len(G.nodes()) < n:
        G.add_node((random(), random()))

    return G

def connect(G):
    """Returns G after executing the following procedure: 1. Select a node X at
    random. 2. Connect X by a straight line to the nearest neighbor Y such that
    X is not already connected to Y and the line crosses no other line.
    3. Repeat 1 and 2 until no more connections are possible.

    """
    nodes = G.nodes()
    while nodes:
        X = choice(nodes)
        Y = nearestValidNeighbor(G, X)
        if Y is not None:
            G.add_edge(X, Y)

        nodes = criticalNodes(G, nodes)

    return G

def criticalNodes(G, nodes):
    """Returns the nodes of G for which possible edges still exist.

    """
    return [node for node in nodes if nearestValidNeighbor(G, node) is not None]

def nearestValidNeighbor(G, X):
    """Returns the nearest neighbor to X such that a straight edge from X to the
    neighbor would cross no other edge. Returns None if no neighbors satisfy the
    condition.

    """
    # Get a list of the nodes sorted by their distance from X
    nodes = G.nodes()
    edges = G.edges()
    nodes.sort(key=lambda x: euclideanDistance(x, X))

    for node in nodes[1:]:
        # Ensure an edge between X and node does not already exist.
        if (X, node) in edges or (node, X) in edges:
            continue

        # Check if an edge between X and node would cross any other edge.
        possibleEdge = (X, node)
        if all(not isIntersecting(edge, possibleEdge)
               for edge in edges):
            return node

    # If we made it here, then no valid edge could be made to a neighbor.
    return None

def euclideanDistance(p, q):
    """Returns the Euclidean distance between te given tuples."""
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def isIntersecting(ef, pq):
    """Returns True if an intersection exists between line segments ef and pq."""
    # credit: http://jsfiddle.net/ytr9314a/4/
    a = list(ef)[0]
    b = list(ef)[1]
    c = list(pq)[0]
    d = list(pq)[1]
    aSide = crossProduct(c, d, a) > 0
    bSide = crossProduct(c, d, b) > 0
    cSide = crossProduct(a, b, c) > 0
    dSide = crossProduct(a, b, d) > 0

    return aSide != bSide and cSide != dSide

def crossProduct(e, f, p):
    return (f[0] - e[0]) * (p[1] - e[1]) - (f[1] - e[1]) * (p[0] - e[0])

def mapProblem(n):
    """Returns a random map-coloring problem represented as a dual graph of n
    vertices.

    """
    return connect(scatter(n))

if __name__ == "__main__":
    main(sys.argv)

# Local Variables:
# flycheck-python-pycompile-executable: "/usr/bin/python2"
# End:
