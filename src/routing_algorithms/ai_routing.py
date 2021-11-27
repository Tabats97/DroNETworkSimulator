
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
