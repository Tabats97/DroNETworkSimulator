
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt
from src.utilities import config
from enum import Enum, auto

class States(Enum):
    WAIT = auto() #I wait even if I have no neighbour and continue my mission
    MOVE = auto() #I move to the depot

class Actions(Enum):
    REMAIN = auto() #self-loop in the state
    CHANGE = auto() #transition from one state to the other


class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)

        self.gamma = 0.9
        self.alfa = 0.1

        self.epsilon = 0.1

        self.waiting = set() #all the packets I am currently waiting for in order to give them to someone else

        self.cur_state = States.WAIT #initial state is set to Wait
        self.Q_values = {(States.WAIT, Actions.REMAIN):0, (States.WAIT, Actions.CHANGE):0,
                        (States.MOVE, Actions.REMAIN):0, (States.MOVE, Actions.CHANGE):0}

    def feedback(self, drone, id_event, delay, outcome):
        if id_event in self.waiting:
            #bad news, I waited and I lost the packet
            if outcome == -1:
                self._updateQ(States.WAIT, Actions.REMAIN, -5000, States.WAIT)
            self.waiting.discard(id_event)

    def relay_selection(self, opt_neighbors, pkd):
        #if self.drone.move_routing: #if I am already returning to the depot then I keep all my packets
        #    return None

        pkd_id = pkd.event_ref.identifier
        
        choice = None
     
        #we use Q_learning to decide what to do when we don't have any neighbours: we wait for one or we move to the depot?
        if len(opt_neighbors) == 0:

            state = self.cur_state
            
            #best action in our current state
            index = np.argmax([self.Q_values[(state, Actions.REMAIN)], self.Q_values[(state, Actions.CHANGE)]])
            action = Actions.REMAIN if index == 0 else Actions.CHANGE

            if (state == States.MOVE and action == Actions.REMAIN) or (state == States.WAIT and action == Actions.CHANGE):
                time_to_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords) / self.drone.speed
                cost = -time_to_depot * 10
                self._updateQ(state, action, cost, States.MOVE)
                self.cur_state = States.MOVE
                choice = -1
            else:
                self.cur_state = States.WAIT #choice is already None
                event_time_to_dead = (self.drone.tightest_event_deadline - self.simulator.cur_step) * self.simulator.time_step_duration
                exp = event_time_to_dead < 300 #self.drone.packet_is_expiring(self.simulator.cur_step)
                if (state == States.MOVE and action == Actions.CHANGE):
                    reward = -100 * self.drone.buffer_length() if exp else 0
                    self._updateQ(state, action, 0, States.WAIT)
                elif pkd_id not in self.waiting:
                    #the reward is delayed for this choice, it can either be positive or negative
                    #depending on if we will find a neighbour for the packet
                    self.waiting.add(pkd_id) #self.waiting[pkd_id] = self.simulator.cur_step
                    
        else:
            best_choice = False
            best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
            for hpk, drone_instance in opt_neighbors:
                #if one of my neighbour is going to the depot, then I give my packets to him
                if hpk.move_to_depot:
                    choice = drone_instance
                    best_choice = True
                    break
                #otherwise I choose the one closest to the depot
                exp_distance = util.euclidean_distance(hpk.cur_pos, self.simulator.depot.coords)
                if exp_distance < best_drone_distance_from_depot:
                    best_drone_distance_from_depot = exp_distance
                    choice = drone_instance
            #good news, I waited and then I found a neighbour for this packet
            if choice is not None and pkd_id in self.waiting:
                reward = 500 if best_choice else 100 #1000 if best_choice else 500 #1/(self.simulator.cur_step - self.waiting[pkd_id]) * 1000
                self._updateQ(States.WAIT, Actions.REMAIN, reward, States.WAIT)
                self.waiting.discard(pkd_id) #note that at the next step I could add this packet again if the transmission was not successful

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
        
