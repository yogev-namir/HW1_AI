import itertools

import check
import search
import random
import math
import json

import utils

ids = ["208253708", "206611477"]


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial["map"]
        initial.pop("map")

        self.distances_dict = {}
        self.build_distances_dictionary()

        self.taxis = []
        for taxi in initial["taxis"]:
            self.taxis.append(taxi)
            taxi = initial["taxis"][taxi]
            taxi["current_fuel"] = taxi["fuel"]
            taxi["current_passengers"] = {}

        initial = json.dumps(initial, indent=4)  # convert to hashable
        search.Problem.__init__(self, initial)

    def build_distances_dictionary(self):
        points = [(i, j) for i in range(len(self.map)) for j in range(len(self.map[0]))]
        for element in itertools.product(points, points):
            p1 = element[0]
            p2 = element[1]
            if self.map[p1[0]][p1[1]] != 'I' and self.map[p2[0]][p2[1]] != 'I':
                self.distances_dict[str(element)] = self.distance(p1, p2)
            else:
                self.distances_dict[str(element)] = float('inf')

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        all_possible_actions = []
        returned_actions = []
        state = json.loads(state)

        for taxi in state["taxis"]:
            all_possible_actions.append(self.actions_per_taxi(state, taxi))

        for element in itertools.product(*all_possible_actions):
            if is_legal(element, state):
                returned_actions.append(element)

        return tuple(returned_actions)

    def actions_per_taxi(self, state, taxi):
        """Returns all the actions that can be executed in the given
                state for a specific taxi."""
        possible_actions = [("wait", taxi)]  # you can always wait

        location = state["taxis"][taxi]["location"]
        i = location[0]
        j = location[1]

        is_fueled = state["taxis"][taxi]["current_fuel"] > 0
        is_not_full = len(state["taxis"][taxi]["current_passengers"]) != state["taxis"][taxi]["capacity"]

        # check if move is possible
        if is_fueled:
            if i != 0 and self.map[i - 1][j] != "I":
                possible_actions.append(("move", taxi, (i - 1, j)))  # move up
            if i != len(self.map) - 1 and self.map[i + 1][j] != "I":
                possible_actions.append(("move", taxi, (i + 1, j)))  # move down
            if j != 0 and self.map[i][j - 1] != "I":
                possible_actions.append(("move", taxi, (i, j - 1)))  # move left
            if j != len(self.map[0]) - 1 and self.map[i][j + 1] != "I":
                possible_actions.append(("move", taxi, (i, j + 1)))  # move right

        # check if pick up is possible
        if is_not_full:
            for passenger in state["passengers"]:
                passenger_location = state["passengers"][passenger]["location"]
                if passenger_location == location:
                    possible_actions.append(("pick up", taxi, passenger))

        # check if drop off is possible
        for passenger in state["taxis"][taxi]["current_passengers"]:
            if location == state["taxis"][taxi]["current_passengers"][passenger]:
                possible_actions.append(("drop off", taxi, passenger))

        # check if refuel is possible
        if self.map[i][j] == "G":
            possible_actions.append(("refuel", taxi))

        return possible_actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = json.loads(state)

        for atomic_action in action:
            act = atomic_action[0]
            taxi = atomic_action[1]

            if act == 'move':
                new_state["taxis"][taxi]["location"] = atomic_action[2]
                new_state["taxis"][taxi]["current_fuel"] -= 1

            if act == 'pick up':
                passenger = atomic_action[2]
                passenger_dest = new_state["passengers"][passenger]["destination"]
                new_state["taxis"][taxi]["current_passengers"][passenger] = passenger_dest
                del new_state["passengers"][passenger]

            if act == 'drop off':
                del new_state["taxis"][taxi]["current_passengers"][atomic_action[2]]

            if act == 'refuel':
                new_state["taxis"][taxi]["current_fuel"] = new_state["taxis"][taxi]["fuel"]

        return json.dumps(new_state)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        is_goal = True
        state = json.loads(state)

        if len(state["passengers"]) != 0:
            is_goal = False

        for taxi in state["taxis"]:
            if len(state["taxis"][taxi]["current_passengers"]) != 0:
                is_goal = False

        return is_goal

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        state = json.loads(node.state)
        arr = []

        for passenger in state["passengers"]:
            minimum = float('inf')
            passenger_loc = state["passengers"][passenger]["location"]
            destination = state["passengers"][passenger]["destination"]
            dist_to_dest = self.distances_dict[str(tuple([tuple(destination), tuple(passenger_loc)]))]

            for taxi in state["taxis"]:
                taxi_location = state["taxis"][taxi]["location"]
                path_len = self.distances_dict[str(tuple([tuple(taxi_location), tuple(passenger_loc)]))] + dist_to_dest
                if path_len == float('inf'):
                    continue
                gas = state["taxis"][taxi]['current_fuel']
                full_fuel = state["taxis"][taxi]['fuel']

                # next, we calculate the total cost assuming there is a gas station everywhere
                minimum = min(minimum, path_len + max(math.ceil((path_len - gas) / full_fuel), 0))

            arr.append(minimum + 2)  # adding the cost of pick_up and drop_off

        for taxi in state["taxis"]:
            taxi_location = state["taxis"][taxi]["location"]
            for passenger in state["taxis"][taxi]["current_passengers"]:
                passenger_dest = state["taxis"][taxi]["current_passengers"][passenger]
                path_len = self.distances_dict[str(tuple([tuple(taxi_location), tuple(passenger_dest)]))]
                if path_len == float('inf'):
                    continue
                gas = state["taxis"][taxi]['current_fuel']
                full_fuel = state["taxis"][taxi]['fuel']

                # next, we calculate the total cost assuming there is a gas station everywhere
                arr.append(path_len + max(math.ceil((path_len - gas) / full_fuel), 0) + 1)

        # consider goal states
        if len(arr) == 0:
            return 0

        return max(arr)  # returns the maximal cost, since taxis work on parallel

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = json.loads(node.state)
        number_of_unpicked_passengers = len(state["passengers"])
        num_taxis = len(state["taxis"])
        number_of_picked_passengers = 0
        for taxi in state["taxis"]:
            number_of_picked_passengers += len(state["taxis"][taxi]["current_passengers"])
        res = (number_of_unpicked_passengers * 2 + number_of_picked_passengers) / num_taxis
        return res

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = json.loads(node.state)
        sum_D = 0
        sum_T = 0
        num_taxis = len(state["taxis"])

        for passenger in state["passengers"]:
            passenger_dict = state["passengers"][passenger]
            sum_D += manhattan_distance(passenger_dict["location"], passenger_dict["destination"])

        for taxi in state["taxis"]:
            taxi_location = state["taxis"][taxi]["location"]
            for passenger in state["taxis"][taxi]["current_passengers"]:
                passenger_dest = state["taxis"][taxi]["current_passengers"][passenger]
                sum_T += manhattan_distance(taxi_location, passenger_dest)

        return (sum_D + sum_T) / num_taxis

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def distance(self, source, destination):
        game = tuple((self.map, source, destination))
        problem = create_shortest_path_problem(game)
        res = search.astar_search(problem, problem.h)
        if res is None:
            return float('inf')
        else:
            return len(search.astar_search(problem, problem.h).path()) - 1


def create_taxi_problem(game):
    return TaxiProblem(game)


def is_legal_aux(action1, action2, state):
    if action1[0] != 'move' and action2[0] != 'move':
        return True
    else:
        if action1[0] == 'move' and action2[0] == 'move':
            return tuple(action1[2]) != tuple(action2[2])
        if action1[0] == 'move':
            return tuple(action1[2]) != tuple(state["taxis"][action2[1]]["location"])
        else:
            return tuple(action2[2]) != tuple(state["taxis"][action1[1]]["location"])


def is_legal(element, state):
    return all([is_legal_aux(action1, action2, state)
                for action1 in element for action2 in element if action1 != action2])


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def create_shortest_path_problem(game):
    return ShortestPathProblem(game)


class ShortestPathProblem(search.Problem):

    def __init__(self, initial):
        self.map = initial[0]  # map
        self.dest = initial[2]  # destination
        initial = tuple(initial[1])  # source
        search.Problem.__init__(self, initial)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        possible_actions = []
        # state = json.loads(state)
        i = state[0]
        j = state[1]
        # move
        if i != 0 and self.map[i - 1][j] != "I":
            possible_actions.append(("move", (i - 1, j)))  # move up
        if i != len(self.map) - 1 and self.map[i + 1][j] != "I":
            possible_actions.append(("move", (i + 1, j)))  # move down
        if j != 0 and self.map[i][j - 1] != "I":
            possible_actions.append(("move", (i, j - 1)))  # move left
        if j != len(self.map[0]) - 1 and self.map[i][j + 1] != "I":
            possible_actions.append(("move", (i, j + 1)))  # move right
        return tuple(possible_actions)

    def result(self, state, action):
        return action[1]  # returns the next location

    def goal_test(self, state):
        return state[0] == self.dest[0] and state[1] == self.dest[1]

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate, which is the Manhattan distance"""
        p1 = node.state
        p2 = self.dest
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
