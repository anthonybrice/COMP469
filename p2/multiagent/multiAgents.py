# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.0

        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), newPos)
            if d <= 1:
                if ghost.scaredTimer > 0:
                    score += 1000
                else:
                    score -= 200

        for capsule in currentGameState.getCapsules():
            d = manhattanDistance(capsule, newPos)
            if d == 0:
                score += 100
            else:
                score += 10.0 / d

        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    d = manhattanDistance((x, y), newPos)
                    if d == 0:
                        score += 100
                    else:
                        score += 1.0 / (d * d)

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        _, bestMove = self.value(gameState, self.depth, self.index)

        return bestMove

    def value(self, state, depth, agent):
        # If we've iterated through every agent
        if agent == state.getNumAgents():
            depth -= 1
            agent = self.index

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agent == self.index:
            return self.maxAct(state, depth)
        else:
            return self.minAct(state, depth, agent)

    def maxAct(self, state, depth):
        v = float("-inf")
        bestMove = None

        for act in state.getLegalActions(self.index):
            score, _ = self.value(state.generateSuccessor(self.index, act),
                                  depth, self.index + 1)

            if score > v:
                v = score
                bestMove = act

        return v, bestMove

    def minAct(self, state, depth, agent):
        v = float("inf")
        bestMove = None

        for act in state.getLegalActions(agent):
            score, _ = self.value(state.generateSuccessor(agent, act), depth,
                             agent + 1)

            if score < v:
                v = score
                bestMove = act

        return v, bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, bestMove = self.value(gameState, float("-inf"), float("inf"),
                                 self.depth, self.index)

        return bestMove

    def value(self, state, alpha, beta, depth, agent):
        # If we've iterated through all agents
        if agent == state.getNumAgents():
            depth -= 1
            agent = self.index

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agent == self.index:
            return self.maxAct(state, alpha, beta, depth)
        else:
            return self.minAct(state, alpha, beta, depth, agent)


    def maxAct(self, state, alpha, beta, depth):
        v = float("-inf")
        move = None

        for act in state.getLegalActions(self.index):
            score, _ = self.value(state.generateSuccessor(self.index, act),
                                  alpha, beta, depth, self.index + 1)
            if score > v:
                v = score
                move = act

            # The pseudo-code in video 12.6 explicitly states this test should
            # be >=. However to pass 8-pacman-game.test, it must be >.
            if v > beta:
                return v, move

            alpha = max(alpha, v)

        return v, move

    def minAct(self, state, alpha, beta, depth, agent):
        v = float("inf")
        move = None

        for act in state.getLegalActions(agent):
            score, _ = self.value(state.generateSuccessor(agent, act), alpha,
                                  beta, depth, agent + 1)
            if score < v:
                v = score
                move = act

            # The pseudo-code in video 12.6 explicitly states this test should
            # be <=. However to pass 6-tied-root.test, it must be <.
            if v < alpha:
                return v, move

            beta = min(beta, v)

        return v, move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, bestMove = self.value(gameState, self.depth, self.index)

        return bestMove

    def value(self, state, depth, agent):
        if agent == state.getNumAgents():
            depth -= 1
            agent = self.index

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agent == self.index:
            return self.maxAct(state, depth)
        else:
            return self.randomAct(state, depth, agent)

    def maxAct(self, state, depth):
        v = float("-inf")
        bestMove = None

        for act in state.getLegalActions(self.index):
            score, _ = self.value(state.generateSuccessor(self.index, act),
                                  depth, self.index + 1)

            if score > v:
                v = score
                bestMove = act

        return v, bestMove

    def randomAct(self, state, depth, agent):
        v = 0.0

        acts = state.getLegalActions(agent)
        p = 1.0 / len(acts)
        for act in acts:
            v += p * self.value(state.generateSuccessor(agent, act), depth,
                                agent + 1)[0]

        return v, None


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = 0
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()

    minDistance = float("inf")
    setMinDistance = False
    for foodPosition in foodPositions:
        foodDistance = util.manhattanDistance(pacmanPosition, foodPosition)
        if foodDistance < minDistance:
            minDistance = foodDistance
            setMinDistance = True

    if setMinDistance:
        score += minDistance

    score += 10 * len(currentGameState.getCapsules())

    for ghostPosition in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(pacmanPosition, ghostPosition)
        if ghostDistance < 2:
            score = float("inf")

    score -= 10 * currentGameState.getScore()

    return -1 * score

# Abbreviation
better = betterEvaluationFunction

# Local Variables:
# flycheck-python-pycompile-executable: "/usr/bin/python2"
# End:
