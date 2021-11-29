
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from enum import Enum, auto

class States(Enum):
    WAIT = auto() 
    MOVE = auto()

class Actions(Enum):
    REMAIN = auto() #self-loop in the state
    CHANGE = auto() #transition from one state to the other


class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        #self.taken_actions = {}  #id event : (old_action)

        self.gamma = 0.9
        self.alfa = 0.1

        self.waiting = {}

        self.cur_state = States.WAIT #initial state is set to Wait
        self.Q_values = {(States.WAIT, Actions.REMAIN):0, (States.WAIT, Actions.CHANGE):0,
                        (States.MOVE, Actions.REMAIN):0, (States.MOVE, Actions.CHANGE):0}

    def feedback(self, drone, id_event, delay, outcome):
        if id_event in self.waiting:
            if outcome == -1:
                self._updateQ(States.WAIT, Actions.REMAIN, -5000, States.WAIT)
            del self.waiting[id_event]

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed

        # Only if you need --> several features:
        # cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
        #                                                width_area=self.simulator.env_width,
        #                                                x_pos=self.drone.coords[0],  # e.g. 1500
        #                                                y_pos=self.drone.coords[1])[0]  # e.g. 500
        # print(cell_index)
        
        #if self.drone.move_routing: #if I am already returning to the depot then I keep all my packets
        #    return None

        pkd_id = pkd.event_ref.identifier
        
        choice = None
     
        #we use Q_learning to decide what to do when we don't have any neighbours: we wait for one or we move to the depot?
        if len(opt_neighbors) == 0:

            state = self.cur_state
            
            index = np.argmax([self.Q_values[(state, Actions.REMAIN)], self.Q_values[(state, Actions.CHANGE)]])
            action = Actions.REMAIN if index == 0 else Actions.CHANGE #best action in our current state
            if (state == States.MOVE and action == Actions.REMAIN) or (state == States.WAIT and action == Actions.CHANGE):
                cost = -1000 #SOSTITUIRE CON COSTO ENERGIA
                self._updateQ(state, action, cost, States.MOVE)
                self.cur_state = States.MOVE
                choice = -1
            else:
                self.cur_state = States.WAIT #choice is already None
                if (state == States.MOVE and action == Actions.CHANGE):
                    #if packet_exp --> -100 (o dipende dal numero di pacchetti nel buffer)
                    self._updateQ(state, action, 0, States.WAIT)
                elif pkd_id not in self.waiting:
                    #the reward is delayed for this choice, it can either be positive or negative
                    #depending on if we will find a neighbour for the packet
                    self.waiting[pkd_id] = self.simulator.cur_step
            
        else:
            best_choice = False
            best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
            for hpk, drone_instance in opt_neighbors:
                #if one of my neighbour is going to the depot, then I give my packets to him
                if hpk.move_to_depot:
                    choice = drone_instance
                    best_choice = True
                    break
                exp_distance = util.euclidean_distance(hpk.cur_pos, self.simulator.depot.coords)
                if exp_distance < best_drone_distance_from_depot:
                    best_drone_distance_from_depot = exp_distance
                    choice = drone_instance
            if choice is not None and pkd_id in self.waiting: #good news, I waited and then I found a neighbour for this packet
                reward = 500 if best_choice else 100 #1/(self.simulator.cur_step - self.waiting[pkd_id]) * 1000
                self._updateQ(States.WAIT, Actions.REMAIN, reward, States.WAIT)
                del self.waiting[pkd_id] #note that at the next step I could add this packet again if the transmission was not successful

        return choice


    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass

    def _updateQ(self, state, action, reward, next_state):
        now = self.Q_values[(state, action)]
        max_next = max([self.Q_values[(next_state, Actions.REMAIN)], self.Q_values[(next_state, Actions.CHANGE)]])    
        self.Q_values[(state, action)] = now + self.alfa * (reward + self.gamma * max_next - now)
        
