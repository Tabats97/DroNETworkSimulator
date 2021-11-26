
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import config

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)

        self.max_delay = self.simulator.event_duration #int: steps, number of time steps that an event lasts  -> to seconds = step * step_duration
        self.n_actions = {}
        self.Q_table = {}

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:",  delay, " - outcome:", outcome)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)

        if id_event in self.taken_actions:
            action = self.taken_actions[id_event]
            del self.taken_actions[id_event]

            # reward = 0
            # if outcome == -1:
            #     reward = (delay/self.max_delay)
            # else:
            #     reward = (1 - (delay/self.max_delay))
            #
            # self.update_Q(action, reward)

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed

        # Only if you need --> several features:

        # cell_index return an index between 1.0 and 16.0 where each index represents a region
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500

        action = None
        neighbors = [drone for _, drone in opt_neighbors]

        # context = (tuple(neighbors), cell_index)
        # key_actions = [key for key in self.Q_table if context == (key[0], key[1])]
        #
        # if key_actions: #AI
        #     if self.rnd_for_routing_ai.uniform(0, 1) < self.epsilon: #epsilon action
        #         action = self.epsilon_action(neighbors)
        #     else:
        #         value_actions = [self.Q_table[k] for k in key_actions] #classical greedy action
        #         action = self.greedy_action(value_actions)
        #
        # else: # not AI

        # if one packet is expiring or the simulation is ending, move to the depot
        if self.drone.packet_is_expiring(self.simulator.cur_step) or self.simulation_ending(self.simulator.cur_step):
            action = -1

        # use georouting based on the y-axis
        else:
            action = self.georouting(opt_neighbors)

        # self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
        # self.drone.residual_energy (that tells us when I'll come back to the depot).
        #  .....

        # Store your current action --- you can add several stuff if needed to take a reward later
        if pkd.event_ref.identifier not in self.taken_actions or self.taken_actions[pkd.event_ref.identifier][-1][-1] != action: #if not same neighbors and region
            self.taken_actions.setdefault(pkd.event_ref.identifier, []).append((tuple(neighbors), cell_index, action))

        # return action:
        # None --> no transmission
        # -1 --> move to depot
        # 0, ... , self.ndrones --> send packet to this drone
        return action  # here you should return a drone object!


    def print(self):
        """
            This method is called at the end of the simulation, can be useful to print some
                metrics about the learning process
        """
        pass

    def georouting(self, opt_neighbors):

        action = None
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)

        for hpk, drone_instance in opt_neighbors:

            # best drone based on vertical waypoint, but we can change this
            if self.drone.waypoint_history and drone_instance.waypoint_history:
                if drone_instance.waypoint_history[-1][1] < self.drone.waypoint_history[-1][1]:
                    action = drone_instance

            # to avoid crash at the beginning
            else:
                exp_position = self.__estimated_neighbor_drone_position(hpk)
                exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.coords)
                if exp_distance < best_drone_distance_from_depot:
                    best_drone_distance_from_depot = exp_distance
                    action = drone_instance

        return action

    def update_Q(self, action, reward):

        n_previous = [self.n_actions[a] + 1 if a in self.n_actions else 1 for a in action]
        self.n_actions.update(dict(zip(action, n_previous)))

        q = [self.Q_table[a] + 1 / self.n_actions[a] * (reward - self.Q_table[a]) if a in self.Q_table else reward for a in action]
        self.Q_table.update(dict(zip(action, q)))

    def epsilon_action(self, neighbors):
        neighbors.append(None)
        drone = self.rnd_for_routing_ai.choice(neighbors)
        return drone

    def greedy_action(self, neighbors, value_actions):
        pass

    def __estimated_neighbor_drone_position(self, hello_message):
        """ estimate the current position of the drone """

        # get known info about the neighbor drone
        hello_message_time = hello_message.time_step_creation
        known_position = hello_message.cur_pos
        known_speed = hello_message.speed
        known_next_target = hello_message.next_target

        # compute the time elapsed since the message sent and now
        # elapsed_time in seconds = elapsed_time in steps * step_duration_in_seconds
        elapsed_time = (self.simulator.cur_step - hello_message_time) * self.simulator.time_step_duration  # seconds

        # distance traveled by drone
        distance_traveled = elapsed_time * known_speed

        # direction vector
        a, b = np.asarray(known_position), np.asarray(known_next_target)
        v_ = (b - a) / np.linalg.norm(b - a)

        # compute the expect position
        c = a + (distance_traveled * v_)

        return tuple(c)

    # if the simulation is ending the packets should be sent to the depot now
    def simulation_ending(self, cur_step):
        time_to_depot = util.euclidean_distance(self.drone.depot.coords, self.drone.coords) / self.drone.speed
        sim_time_to_dead = (self.simulator.len_simulation - cur_step) * self.simulator.time_step_duration
        return sim_time_to_dead - 5 < time_to_depot <= sim_time_to_dead
