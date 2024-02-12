import search
import random
import math
import json
import itertools
import numpy as np

ids = ["212984801", "316111442"]


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial["map"]
        self.distances_matrix, self.gas_locations = self.floyd_warshall()
        search.Problem.__init__(self, self.add_data_and_transform_to_json(initial))

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        state = json.loads(state)  # Transforming the state from json string to dictionary
        possible_actions = {}
        for taxi in state["taxis"]:
            possible_actions[taxi] = []
            # Checking if the taxi can move in the grid.
            if state["taxis"][taxi]["current_fuel"] > 0:
                possible_actions[taxi] = possible_actions[taxi] + self.check_possible_grid_moves(state["taxis"][taxi]["location"], taxi)

            # Checking if the taxi can pick up a passenger.
            if state["taxis"][taxi]["current_capacity"] < state["taxis"][taxi]["capacity"]:
                possible_actions[taxi] = possible_actions[taxi] + self.check_if_contains_passenger(state["taxis"][taxi]["location"], state["passengers"], taxi, state["taxis"][taxi]["passengers"])

            # Checking if the taxi can drop off a passenger.
            possible_actions[taxi] = possible_actions[taxi] + self.check_drop_off_passenger(state["taxis"][taxi]["location"], state["taxis"][taxi]["passengers"], taxi, state["passengers"])

            # Checking if the taxi can refuel.
            if self.map[state["taxis"][taxi]["location"][0]][state["taxis"][taxi]["location"][1]] == 'G':
                possible_actions[taxi] = possible_actions[taxi] + [("refuel", taxi)]

            possible_actions[taxi] = possible_actions[taxi] + [("wait", taxi)]

        all_actions = tuple(itertools.product(*list(possible_actions.values())))
        all_actions = self.eliminate_not_valid_actions(all_actions, state)

        return all_actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = json.loads(state)  # Transforming the state from json string to dictionary.
        for taxi_action in action:
            if taxi_action[0] == "wait":
                continue

            elif taxi_action[0] == "move":
                # Move the taxi, update its location and decrease its fuel by 1 unit.
                state["taxis"][taxi_action[1]]["location"] = taxi_action[2]
                state["taxis"][taxi_action[1]]["current_fuel"] -= 1
                # Update the passenger's location.
                for passenger in state["taxis"][taxi_action[1]]["passengers"]:
                    state["passengers"][passenger]["location"] = taxi_action[2]

            elif taxi_action[0] == "pick up":
                # Adding the passenger to the taxi passengers list, increasing the taxi capacity by 1.
                state["taxis"][taxi_action[1]]["passengers"].append(taxi_action[2])
                state["taxis"][taxi_action[1]]["current_capacity"] += 1
                state["passengers"][taxi_action[2]]["picked_up"] = True

            elif taxi_action[0] == "drop off":
                # Removing the passenger from the taxi's passengers' list, decreasing the taxi capacity by 1.
                state["taxis"][taxi_action[1]]["passengers"].remove(taxi_action[2])
                state["taxis"][taxi_action[1]]["current_capacity"] -= 1
                state["num_not_reached_to_dest"] -= 1
                state["passengers"][taxi_action[2]]["dropped_off"] = True

            elif taxi_action[0] == "refuel":
                # Updating taxi's current fuel to its maximum.
                state["taxis"][taxi_action[1]]["current_fuel"] = state["taxis"][taxi_action[1]]["fuel"]

        state = json.dumps(state, sort_keys=True)  # Transforming the state from dictionary to json string.
        return state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state = json.loads(state)
        if state["num_not_reached_to_dest"] != 0:
            return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        # Initialize variables.
        state = json.loads(node.state)
        n = len(self.map[0])
        taxis_num = len(state["taxis"])

        # If all passengers dropped off, then we've reached solution.
        if state["num_not_reached_to_dest"] == 0:
            return 0

        # Find and exclude branches that do not lead to solution.
        for taxi in state["taxis"]:
            gas_unreachable = True
            for gas_location in self.gas_locations:  # Search for a close enough gas station.
                if self.calc_distance(state["taxis"][taxi]["location"], gas_location, n) <= \
                        state["taxis"][taxi]["current_fuel"]:
                    gas_unreachable = False

            if taxis_num == 1:
                for passenger in state["passengers"].values():
                    if gas_unreachable:
                        # If taxi can't reach passenger nor a gas station, then there is no solution.
                        if passenger["picked_up"] is False and \
                                self.calc_distance(state["taxis"][taxi]["location"], passenger["location"], n) > \
                                state["taxis"][taxi]["current_fuel"] and gas_unreachable:
                            return math.inf

                        # If taxi can't reach passenger's destination nor a gas station, then there is no solution.
                        elif passenger["dropped_off"] is False and \
                                self.calc_distance(state["taxis"][taxi]["location"], passenger["destination"], n) > \
                                state["taxis"][taxi]["current_fuel"] and gas_unreachable:
                            return math.inf

            else:
                for passenger in state["taxis"][taxi]["passengers"]:
                    if gas_unreachable:
                        # If taxi can't reach passenger's destination nor a gas station, then there is no solution.
                        if self.calc_distance(state["taxis"][taxi]["location"],
                                              state["passengers"][passenger]["destination"], n) > \
                                state["taxis"][taxi]["current_fuel"]:
                            return math.inf

            if state["taxis"][taxi]["current_fuel"] == 0 and self.map[state["taxis"][taxi]["location"][0]][
                state["taxis"][taxi]["location"][1]] != "G":
                # If only one taxi exists, and it ran out of fuel before all passengers reached their destination,
                # then there is no solution.
                if taxis_num == 1 and state["taxis"][taxi]["current_capacity"] == 0 and \
                        state["num_not_reached_to_dest"] > 0:
                    return math.inf
                else:
                    for passenger in state["taxis"][taxi]["passengers"]:
                        # If one of the taxis ran out of fuel before delivering all of its passengers to their
                        # destination, then there is no solution.
                        if state["passengers"][passenger]["location"] != state["passengers"][passenger]["destination"]:
                            return math.inf

        return max(self.h_1(node), self.h_3(node))

    def h_3(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        state = json.loads(node.state)

        # Initialize variables.
        n = len(self.map[0])
        taxis_num = len(state["taxis"])
        h = 0

        furthest_passenger = []
        max_distance = 0
        # Search for passengers with the biggest distance from their location to their destination.
        for passenger in state["passengers"]:
            if state["passengers"][passenger]["dropped_off"] is False:
                if max_distance < self.calc_distance(state["passengers"][passenger]["location"],
                                                     state["passengers"][passenger]["destination"], n):
                    furthest_passenger.clear()
                    furthest_passenger.append(passenger)
                    max_distance = self.calc_distance(state["passengers"][passenger]["location"],
                                                      state["passengers"][passenger]["destination"], n)
                elif max_distance == self.calc_distance(state["passengers"][passenger]["location"],
                                                        state["passengers"][passenger]["destination"], n):
                    furthest_passenger.append(passenger)

        # Find within the passengers that were find above the "most expensive" passenger, i.e. the one that costs
        # the biggest number of moves.
        steps = []
        for furthest in furthest_passenger:
            taxi_pass_distance = math.inf
            fuel = 0

            for taxi in state["taxis"]:
                if taxi_pass_distance > self.calc_distance(state["taxis"][taxi]["location"],
                                                           state["passengers"][furthest]["location"], n):
                    best_taxi = taxi
                    taxi_pass_distance = self.calc_distance(state["taxis"][taxi]["location"],
                                                            state["passengers"][furthest]["location"], n)
                elif taxi_pass_distance == self.calc_distance(state["taxis"][taxi]["location"],
                                                              state["passengers"][furthest]["location"], n):
                    if fuel < state["taxis"][taxi]["fuel"]:
                        best_taxi = taxi
                        fuel = state["taxis"][taxi]["fuel"]

            # Check if the most expensive passenger is already on a taxi or not, and check if the taxi will need to
            # refuel in order to reach its destination, and add number of moves accordingly.
            moves_num = max_distance + taxi_pass_distance
            if state["taxis"][best_taxi]["fuel"] < moves_num:
                if taxis_num == 1:
                    if state["passengers"][furthest]["picked_up"] is True:
                        steps.append(2 + moves_num + (moves_num - state["taxis"][best_taxi]["current_fuel"] - 1) //
                                     state["taxis"][best_taxi]["fuel"])
                    else:
                        steps.append(3 + moves_num + (moves_num - state["taxis"][best_taxi]["current_fuel"] - 1) //
                                     state["taxis"][best_taxi]["fuel"])
                else:
                    if state["passengers"][furthest]["picked_up"] is True:
                        steps.append(moves_num + 2)
                    else:
                        steps.append(moves_num + 3)
            else:
                if state["passengers"][furthest]["picked_up"] is True:
                    steps.append(moves_num + 1)
                else:
                    steps.append(moves_num + 2)

        h += max(steps)
        return h

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = json.loads(node.state)
        unpicked_passengers = 0
        picked_but_undelivered = 0
        num_taxis = len(state["taxis"])
        for passenger in state["passengers"]:
            if not state["passengers"][passenger]["picked_up"]:
                unpicked_passengers += 1
            elif not state["passengers"][passenger]["dropped_off"]:
                picked_but_undelivered += 1
        return (unpicked_passengers * 2 + picked_but_undelivered)/num_taxis

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = json.loads(node.state)
        num_taxis = len(state["taxis"])
        sum_D_i = 0
        sum_T_i = 0
        for passenger in state["passengers"]:
            if not state["passengers"][passenger]["picked_up"]:
                sum_D_i += self.manhattan_distance(state["passengers"][passenger]["initial_location"],
                                                   state["passengers"][passenger]["destination"])
            elif not state["passengers"][passenger]["dropped_off"]:
                sum_T_i += self.manhattan_distance(state["passengers"][passenger]["location"],
                                                   state["passengers"][passenger]["destination"])
        return (sum_D_i + sum_T_i)/num_taxis

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""
    def add_data_and_transform_to_json(self, initial):
        """
        adding data to the state and removing the map
        """""
        del(initial["map"])
        # Adding data about the taxis
        for taxi in initial["taxis"].keys():
            initial["taxis"][taxi]["passengers"] = []
            initial["taxis"][taxi]["current_fuel"] = initial["taxis"][taxi]["fuel"]
            initial["taxis"][taxi]["current_capacity"] = 0

        # Adding data about the passengers
        initial["num_not_reached_to_dest"] = len(initial["passengers"])
        for passenger in initial["passengers"].keys():
            initial["passengers"][passenger]["picked_up"] = False
            initial["passengers"][passenger]["dropped_off"] = False
            initial["passengers"][passenger]["initial_location"] = initial["passengers"][passenger]["location"]

        state = json.dumps(initial, sort_keys=True)
        return state

    def check_possible_grid_moves(self, location, taxi_name):
        """
        check the possible moves in the grid
        """
        possible_moves = []
        x = location[0]
        y = location[1]
        if x > 0:
            possible_moves.append(("move", taxi_name, (x-1, y))) if self.map[x-1][y] != 'I' else None
        if x < len(self.map)-1:
            possible_moves.append(("move", taxi_name, (x+1, y))) if self.map[x+1][y] != 'I' else None
        if y > 0:
            possible_moves.append(("move", taxi_name, (x, y-1))) if self.map[x][y-1] != 'I' else None
        if y < len(self.map[0])-1:
            possible_moves.append(("move", taxi_name, (x, y+1))) if self.map[x][y+1] != 'I' else None

        return possible_moves

    def check_if_contains_passenger(self, location, passengers, taxi_name, taxi_passengers):
        """
        check if the location contains a passenger and if the taxi doesn't already have him and the passenger hasnt reached his destination
        """
        passenger_actions = []
        for passenger in passengers.keys():
            if passengers[passenger]["location"] == location and passenger not in taxi_passengers and not passengers[passenger]["dropped_off"]:
                passenger_actions.append(("pick up", taxi_name, passenger))
        return passenger_actions

    def check_drop_off_passenger(self, location, passengers_of_taxi, taxi_name, all_passengers):
        """
        check if the location contains a passenger
        """
        passenger_actions = []
        for passenger in passengers_of_taxi:
            if all_passengers[passenger]["destination"] == location:
                passenger_actions.append(("drop off", taxi_name, passenger))
        return passenger_actions

    def extract_locations(self, action, state):
        """
        extract the locations from the action
        """
        locations = []
        for taxi_action in action:
            if taxi_action[0] == "move":
                locations.append(tuple(taxi_action[2]))
            else:
                locations.append(tuple(state["taxis"][taxi_action[1]]["location"]))

        return locations

    def eliminate_not_valid_actions(self, all_actions, state):
        """
        check all actions and eliminate the ones that are not valid (2 taxis in the same location)
        """
        new_all_actions = []
        if len(state["taxis"]) > 1:
            for action in all_actions:
                locations = self.extract_locations(action, state)
                if len(locations) == len(set(locations)):
                    new_all_actions.append(action)
            return new_all_actions
        return all_actions

    def calc_distance(self, x, y, n):
        return self.distances_matrix[n*x[0] + x[1]][n*y[0] + y[1]]

    def floyd_warshall(self):
        size = len(self.map[0]) * len(self.map) # Get number of vertices in grid.
        matrix = [[math.inf for _ in range(size)] for _ in range(size)] # initialize distances matrix.
        gas_locations = []

        # Build first distances according to neighbouring vertices.
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == "G":
                    gas_locations.append((i, j))
                if self.map[i][j] == "I":
                    continue
                else:
                    matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j] = 0
                    if 0 < i < len(self.map) - 1 and 0 < j < len(self.map[0]) - 1:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

                    elif i == 0 and j == 0:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1

                    elif i == 0 and j < len(self.map[0]) - 1:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

                    elif i == 0 and j == len(self.map[0]) - 1:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

                    elif i < len(self.map) - 1 and j == 0:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1

                    elif i == len(self.map) - 1 and j == 0:
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1

                    elif i == len(self.map) - 1 and j == len(self.map[0]) - 1:
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

                    elif i < len(self.map) - 1 and j == len(self.map[0]) - 1:
                        if self.map[i + 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i + 1) * len(self.map[0]) + j] = 1
                            matrix[(i + 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

                    elif i == len(self.map) - 1 and j < len(self.map[0]) - 1:
                        if self.map[i - 1][j] != "I":
                            matrix[i * len(self.map[0]) + j][(i - 1) * len(self.map[0]) + j] = 1
                            matrix[(i - 1) * len(self.map[0]) + j][i * len(self.map[0]) + j] = 1
                        if self.map[i][j + 1] != "I":
                            matrix[i * len(self.map[0]) + j + 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j + 1] = 1
                        if self.map[i][j - 1] != "I":
                            matrix[i * len(self.map[0]) + j - 1][i * len(self.map[0]) + j] = 1
                            matrix[i * len(self.map[0]) + j][i * len(self.map[0]) + j - 1] = 1

        # Finding optimal distances via floyd warshall algorithm.
        # Taking all vertices one by one and setting them as intermediate vertices
        for k in range(size):
            # Pick all vertices as source one by one.
            for s in range(size):
                # Pick all vertices as the destination for the above chosen source vertex.
                for r in range(size):
                    # Update the value of matrix[i][j] if k provides a shortest path from i to j
                    matrix[s][r] = min(matrix[s][r], matrix[s][k] + matrix[k][r])

        return matrix, gas_locations

    def manhattan_distance(self, location1, location2):
        """
        calculate the manhattan distance between two locations
        """
        return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])


def create_taxi_problem(game):
    return TaxiProblem(game)
