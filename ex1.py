import search
import random
import math
from search import Problem
from itertools import product
import copy
from utils import manhattan_distance, create_inverse_dict

# TODO add all non static functions to utils file
ids = ["318880754", "324079763"]
infinity = float('inf')
FULL = 2
"""
        {
            "map": [
                ['S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S'],
                ['B', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S']
            ],
            "pirate_ships": {"pirate_ship_1": (2, 0)},
            "treasures": {'treasure_1': (0, 2)},
            "marine_ships": {'marine_1': [(1, 1), (1, 2), (2, 2), (2, 1)]}
        },

"""
class OnePieceProblem(Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial): # avishag + yogev
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.map = initial['map']
        #self.pirate_ships = initial['pirate_ships']
        self.treasures = initial['treasures']
        self.inverse_treasures = create_inverse_dict(initial['pirate_ships'])
        self.marine_ships = initial['marine_ships']
        self.num_rows = len(self.map) # list of lists
        self.num_cols = len(self.map[0]) # list
        self.num_treasures = len(self.treasures)
        self.num_pirate_ships = len(self.pirate_ships)
        self.symbols_dict = self.symbols_to_dict() # has 'I' 'S' 'B' 'Adj' 'T' keys with corresponding positions on map
        # Insert here a distance matrix (?)
        # Insert here a list of the treasures locations (?)

        #create initial and goal states
        initial = State(initial) 
        search.Problem.__init__(self, initial)
    
    def actions(self, state): #yogev 
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions 
        as defined in the problem description file"""
        valid_actions = {}

        for pirate_ship, location in state.pirate_ships.items():
            valid_actions[pirate_ship] = {}

            valid_actions[pirate_ship]['sail'] = self.check_sail(pirate_ship, location)
            valid_actions[pirate_ship]['collect_treasure'] = self.check_collect_treasure(state, pirate_ship, location)
            valid_actions[pirate_ship]['deposit_treasure'] = self.check_deposit_treasure(state, pirate_ship, location)
            valid_actions[pirate_ship]['wait'] = self.check_wait(state, pirate_ship)

        all_actions = []
        for pirate_ship, pirate_ship_dict in valid_actions.items():
            pirate_ship_actions = []
            for _, actions in pirate_ship_dict.items(): 
                if len(actions) > 0: # Non empty action list
                    pirate_ship_actions += actions
            all_actions.append(pirate_ship_actions)
        all_actions_tuples = list(product(*all_actions))
        return all_actions_tuples
        
        '''
        all_actions = tuple(itertools.product(*list(possible_actions.values()))) # ((),(),())
        all_actions = self.eliminate_not_valid_actions(all_actions, state)   
        '''    

   
    def check_sail(self, pirate_ship, location):
        row, col = location
        directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        valid_actions = list()

        for _, (d_row, d_col) in directions.items():
            new_row, new_col = row + d_row, col + d_col
            new_location = (new_row, new_col)
            
            if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols and (new_location not in self.symbols_dict['I']):
                valid_actions.append(("sail", pirate_ship, new_location))

        return valid_actions
        
    def check_collect_treasure(self, state, pirate_ship, location):
        # TODO: check inventory (pirate_ships_capacity)
        valid_action = list()
        pirate_ship_can_collect = state.pirate_ships_capacity[pirate_ship] != FULL # Boolean
        treasures_collected = state.treasures_collected
        
        if pirate_ship_can_collect and (location in self.symbols_dict['Adj']):
            treasures_names = self.inverse_treasures[location]
            for name in treasures_names:
                if (treasures_collected[name]) or (name in state.pirate_ships_load[pirate_ship]): # treasure was already deposited 
                    continue
                valid_action.append(("collect_treasure", pirate_ship, name))
        return valid_action
    
    def check_deposit_treasure(self, state, pirate_ship, location):
        valid_action = []
        inventory = state.pirate_ships_capacity[pirate_ship]
        
        if (location in self.symbols_dict['B']) and (inventory > 0):
            valid_action.append(('deposite_treasure', pirate_ship))
        return valid_action

    def check_wait(self, pirate_ship):
        valid_actions = list(('wait', pirate_ship))
        return valid_actions


    def result(self, state, action): #avishag 
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        
        new_state = copy.deepcopy(state) #check if it really deep copy it
        
        for sub_action in action:
            pirate_ship = sub_action[1] 
        
            if sub_action[0] == "sail":
                new_location = sub_action[2]
                new_state.update_sail(pirate_ship, new_location)
                new_state.update_crash(pirate_ship)

            elif sub_action[0] == "collect_treasure":
                treasure_name = sub_action[2]
                new_state.update_collect_treasure(pirate_ship, treasure_name)
                new_state.update_crash(pirate_ship)

            elif sub_action[0] == "deposite_treasure": # crash with a marine doesnt metter at this point
                new_state.update_deposite_treasure(pirate_ship)

            else :
                new_state.update_crash(pirate_ship)    
            
        new_state.update_other()    
        return new_state
                
    
    def goal_test(self, state): # done
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return False if state.num_treasures_left_to_collect else True
    

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        pass

    """
        def h_max(self, node):
            Max of h_1 and h_2
            return max(self.h_1(node), self.h_2(node))

        def h_weighted_sum(self, node, alpha=0.5, beta=0.5):
            Weighted sum of h_1 and h_2 with weights alpha and beta
            return alpha * self.h_1(node) + beta * self.h_2(node)

        def h_average(self, node):
            Average of h_1 and h_2
            return (self.h_1(node) + self.h_2(node)) / 2

        def h_nonlinear(self, node):
            Example non-linear combination: square root of the sum of squares
            return (self.h_1(node)**2 + self.h_2(node)**2)**0.5
    """
    
    def h_1(self, node): #done
        return self.num_treasures / self.num_pirate_ships

    def h_2(self, node): #done
        # returns sum of dintances for the closest sea cell adjacent to a treasure for each treasure
        if len(self.symbols_dict['T']) - len(self.symbols_dict['RT']) != 0: #there exists an unreachable treasure
            return infinity
        
        reachable_treasure_locs = self.symbols_dict['RT'] # 'RT' : [(i,j) for every position of an island with a treasure]
        base_loc = self.symbols_dict['B'][0] # base location
        sum_dist = sum([manhattan_distance(base_loc, treasure_loc) - 1 for treasure_loc in reachable_treasure_locs]) # huristic dist to adj to tresure
        return sum_dist / self.num_pirate_ships

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def symbols_to_dict(self): #done
        map = self.map
        num_rows = self.num_rows
        num_cols = self.num_cols
        symbols_dict = {symbol : [] for symbol in ['S', 'I', 'B', 'Adj', 'T', 'RT']}

        for row,col in itertools.product(range(num_rows), range(num_cols)): #every cell
            symbols_dict[ map[row][col] ].append((row,col)) # every symbol holds a list of all of the idx it appears at

        symbols_dict['T'] = [treasure_loc for treasure_name, treasure_loc in self.treasures.items()]

        for row ,col in symbols_dict['I'] : # for every 'T' position (i,j) when counting each adj cell once
            
            if (row, col) in symbols_dict['T']: # only if this island has a treasure
            
                for adj_row, adj_col in [(row -1, col),(row , col -1),(row, col +1),(row+1, col)]: # all adj cells (even if not exists)
                
                    if 0<= adj_row < num_rows and 0<= adj_col < num_cols: #if the cell exists
                    
                        if (adj_row,adj_col) in symbols_dict['S']: #its a sea cell

                            symbols_dict['Adj'].append((adj_row, adj_col))

        
        for row, col in self.symbols_dict['T']: # specific treasure

            for adj_row, adj_col in [(row -1, col),(row , col -1),(row, col +1),(row+1, col)]: # all adj cells (even if not exists)

                if 0<= adj_row < self.num_rows and 0<= adj_col < self.num_cols: #if the cell exists

                    if (row,col) in self.symbols_dict['S']: #its a sea cell

                        self.symbols_dict['RT'].append((row, col)) #only reachable treasures

        return symbols_dict
            

def create_onepiece_problem(game):
    return OnePieceProblem(game)

"""
    The following methods are general methods and not related directly to the class OnePieceProblem
"""

class State:
    # DONT FORGET TO MAKE IT HASHABLE
        def __init__(self, initial): # state is pirate and marine ships current positions, #treasures left to collect and their positions, and the current capacity of the pirate ships
            self.initial = initial # the fixed game

            self.pirate_ships_positions = initial['pirate_ships']
            self.pirate_ships_capacity = {pirate_ship : 0 for pirate_ship in self.pirate_ships_positions.keys()}
            self.pirate_ships_load = {pirate_ship : [] for pirate_ship in self.pirate_ships_positions.keys()}

            treasures = initial['treasures']
            self.treasures_collected = {treasure : False for treasure in treasures.keys()}
            self.num_treasures_left_to_collect = len(initial['treasures'])

            self.marine_ships_paths = initial['marine_ships']
            self.marine_ships_positions = {marine_ship : (path[0],1) for marine_ship, path in self.marine_ships_paths.items()}


        def __deepcopy__(self, memo):
            new_initial = copy.deepcopy(self.initial, memo)
            new_state = State(new_initial)

            new_state.pirate_ships_positions = copy.deepcopy(self.pirate_ships_positions, memo)
            new_state.pirate_ships_capacity = copy.deepcopy(self.pirate_ships_capacity, memo)
            new_state.pirate_ships_load = copy.deepcopy(self.pirate_ships_load, memo)

            new_state.treasures_collected = copy.deepcopy(self.treasures_collected, memo)
            new_state.num_treasures_left_to_collect = self.num_treasures_left_to_collect # deep copy ?

            new_state.marine_ships_positions = copy.deepcopy(self.marine_ships_positions)

            return new_state
        
        def __eq__(self, other): #define equality
            if not isinstance(other, State):
                return NotImplemented
            
            dict_attributes = [ # dictionary attributes of the class to be compared (whos relevant for state definition)
            'pirate_ships_positions', 
            'pirate_ships_capacity',
            'pirate_ships_load',
            'treasures_collected', 
            'marine_ships_positions'
            ]
    
            for attr in dict_attributes: # deeply compares the dictionaries
                if getattr(self, attr, None) != getattr(other, attr, None):
                    return False

            if self.num_treasures_left_to_collect !=  other.num_treasures_left_to_collect :
                return False
            
            return True

        
        def __hash__(self): # TODO: FINISH
            '''
            def __hash__(self):
            # Example: Compute the hash based on immutable properties. This is a placeholder and may need adjustment.
            # Be cautious as making instances hashable and using them as dictionary keys or in sets requires that
            # instances are immutable for the hash to be consistent.
            return hash((tuple(sorted(self.treasures.items())), tuple(sorted(self.marine_ships_positions.items())))) #its not good - just a chatgpy option
            '''
            # Compute a hash using hashable attributes
            return hash((tuple(sorted(self.treasures_collected.items())), 
                        tuple(sorted(self.pirate_ships_capacity.items())), 
                        tuple(sorted(self.marine_ships_positions.items())),
                        self.num_treasures_left_to_collect))
        
        
        def update_sail(self, pirate_ship, new_location): 
            self.pirate_ships_positions[pirate_ship] = new_location
            return

        def update_collect_treasure(self, pirate_ship, treasure_name):
            self.pirate_ships_capacity[pirate_ship] += 1 # im not changing treasures collected here as it is not sure that the ship would arrive safly to base
            self.pirate_ships_load[pirate_ship].append(treasure_name)
            return

        def update_deposite_treasure(self, pirate_ship):
            pirate_ship_load = self.pirate_ships_load[pirate_ship]
            
            for treasure_name in pirate_ship_load:
                if not self.treasures_collected[treasure_name]: # it is the first time collecting this treasure
                    self.treasures_collected[treasure_name] = True
                    self.num_treasures_left_to_collect -=1

            self.unload(pirate_ship)

        def update_marine_ships(self): # update marine ships position in the patrol path
            for marine_ship, path in self.marine_ships_paths.items():
                if len(path) == 1: continue # stationary marine ship (single position in its path) 
                current_position, direction = self.marine_ships_positions[marine_ship] # (position, direction)
     
                if current_position in path:
                    current_position_index = path.index(current_position)
                    if (current_position_index == len(path)-1 and direction == 1) or (current_position_index == 0 and direction == -1):
                        self.marine_ships_positions[marine_ship][1] = (-1)*direction
                else:
                    print(f"Error: {marine_ship} is in an invalid position {current_position}\n")
                    print(f'Valid path is {path}')
                self.marine_ships_positions[marine_ship] = path[current_position_index + direction]

        def update_crash(self, pirate_ship):
            marine_ships_current_positions = [value[0] for value in self.marine_ships_positions.values()] # [(),()....]
            pirate_ship_position = self.pirate_ships_positions[pirate_ship]
            if pirate_ship_position in marine_ships_current_positions:
                self.unload(pirate_ship)

        def unload(self, pirate_ship):
            self.pirate_ships_capacity[pirate_ship] = 0
            self.pirate_ships_load[pirate_ship] = []

# TODO hash function
# TODO h
# TODO implement A*                        