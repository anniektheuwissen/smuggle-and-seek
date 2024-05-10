import mesa
import numpy as np
from more_itertools import powerset

from .container import Container

"""
Agent class: the customs agent and smuggler agent inherit from this class
"""
class Agent(mesa.Agent):
    def __init__(self, unique_id, model, tom_order, learning_speed):
        """
        Initializes the agent 
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed: The speed at which the agent learns
        """
        super().__init__(unique_id, model)
        self.tom_order = tom_order
        self.points = 0
        self.points_queue = [0] * 50
        self.action = []
        self.failed_actions = []
        self.succes_actions = []

        # Define all possible actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(list, powerset(np.arange(num_cont))))[1:]

        # Initialize learning speed, belief vectors, value phi, prediction and confidence needed for tom_reasoning
        self.learning_speed = learning_speed
        self.b0 = np.array([1/num_cont] * num_cont)
        self.b1 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont-1)
        self.simulation_phi = np.zeros(len(self.b1))
        self.prediction_a1 = np.zeros(num_cont)
        self.c1 = 0

    def merge_prediction(self):
        """
        Merges prediction with its belief b0
        """
        W = np.zeros(len(self.b1))
        for c in range(len(self.b1)):
            W[c] = self.c1 * self.prediction_a1[c] + (1-self.c1) * self.b0[c]
        return W

    def common_features(self, c, cstar):
        """
        Returns the amount of common features that container c and container cstar have
        """
        container_c = self.model.get_agents_of_type(Container)[c]; container_cstar = self.model.get_agents_of_type(Container)[cstar]
        return 0 + (container_c.features["cargo"] == container_cstar.features["cargo"]) + (container_c.features["country"] == container_cstar.features["country"]) 

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        pass

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        pass

    def step_tom2(self):
        """
        Chooses an action associated with second-order theory of mind reasoning
        """
        pass

    def take_action(self):
        """
        Performs chosen action
        """
        pass
    
    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        and taking this action
        """
        # Choose action based on order of tom reasoning
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        elif self.tom_order == 2: self.step_tom2()
        else: print("ERROR: Agent cannot have a theory of mind reasoning above the second order")

        # Take action
        self.take_action()