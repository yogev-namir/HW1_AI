"""Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""

from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, Stack, FIFOQueue, PriorityQueue, name,
    distance
)

from collections import defaultdict
import math
import random
import sys
import bisect
import time

infinity = float('inf')

# ______________________________________________________________________________

class Problem(object):
    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        # self.depth = 0
        # if parent:
        #     self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state  # ?

    def expand(self, problem):
        """List the nodes reachable in one step from this node using map."""
        return list(map(lambda action: self.child_node(problem, action), problem.actions(self.state))) # here i get all the succesor nodes (creating all of them a state)

    def child_node(self, problem, action): # here i use result that (that would create a result state for each state possible )
        """[Figure 3.10]"""
        next = problem.result(self.state, action)  # this is a state, this state should be a jason!
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ______________________________________________________________________________
def best_first_graph_search(problem, f):
    start_time = time.time()  # Start timing

    f = memoize(f, 'f')  # Precompute f values for efficiency
    root_node = Node(problem.initial)

    init_time = time.time()  # Check time after initialization

    if problem.goal_test(root_node.state):
        return root_node

    open_list = PriorityQueue(min, f)  # Renamed for clarity
    open_list.append(root_node)
    closed_set = set()  # States we have visited
    distance = {root_node.state: 0}  # Tracks the best-known distance to each state

    setup_time = time.time()  # Check time after setup

    loop_start_time = time.time()  # Start timing the loop

    while open_list:
        current_node = open_list.pop()  # now it is the next iteration!!! and i got all the new nodes 
        # here i create a state out of the jason!!!! state = State(jason) #which is realy initial. mabey i need to copy the shape of initial and that would be my state

        
        if current_node.state in closed_set and current_node.path_cost >= distance[current_node.state]:
            continue  # Skip if we've found a better path already

        closed_set.add(current_node.state) 
        distance[current_node.state] = current_node.path_cost

        if problem.goal_test(current_node.state):
            return current_node
        

        # we will produce node node and than insert the node into the open_list so the h is calculated for it and it is inserted into the queue with the right h and than
        # the next noide would be created and use the same memmoey (they share state, and we would always hols the trouth value so that the node would know whats was the state initialy)

        for successor in current_node.expand(problem): ##############expend now i go for each successor
            if successor.state not in distance or successor.path_cost < distance[successor.state]:
                open_list.append(successor) ######### i put all the new nodes in the open list - need a check
                distance[successor.state] = successor.path_cost

    loop_end_time = time.time()  # End timing the loop
    print(f"Loop iteration time: {loop_end_time - loop_start_time}")

    end_time = time.time()  # End timing
    print(f"Initialization time: {init_time - start_time}")
    print(f"Setup time: {setup_time - init_time}")
    print(f"Total time: {end_time - start_time}")

    return None



"""
def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    root_node = Node(problem.initial)
    if problem.goal_test(root_node.state):
        return root_node
    open = PriorityQueue(min, f)
    open.append(root_node) # contains nodes
    closed = set() # contains states
    distance = {root_node.state: 0}
    while not open.empty():
        min_node = open.pop()
        for succesor in min_node.expand(problem):
            successor_cost = distance[min_node.state] + succesor.path_cost
            if succesor.state not in closed and successor_cost < distance[succesor.state]:
                distance[succesor.state] = successor_cost





        if (min_node.state not in closed) or (min_node.path_cost < distance[min_node.state]):
            closed.add(min_node.state)
            if problem.goal_test(min_node.state):
                return min_node
            for succsesor in min_node.expand(problem):
                if f(succsesor) < infinity: 
                    open.insert(succsesor)
    return None
"""


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    # Memoize the heuristic function for better performance
    h = memoize(h or problem.h, 'h')

    # Function to calculate f(n) = g(n) + h(n)
    # Memoize this function for better performance
    f = memoize(lambda n: n.path_cost + h(n), 'f')

    # TODO: Implement the rest of the A* search algorithm

    return best_first_graph_search(problem, lambda node: node.path_cost + h(node)) ########## heres f !!! and i calculate h on the node when it already exists.. do i calculate all of the h together ?

