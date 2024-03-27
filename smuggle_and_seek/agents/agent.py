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
        Initializes the agent Customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        """
        super().__init__(unique_id, model)
        self.tom_order = tom_order
        self.points = 0
        self.action = []
        self.failed_actions = []
        self.succes_actions = []

        # Define all possible actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(list, powerset(np.arange(num_cont))))[1:]

        # Initialize learning speed, belief vectors and subjective value needed for tom_reasoning
        self.learning_speed = learning_speed
        self.b0 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont-1)

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
    
    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        and taking this action
        """
        pass
        
    def update_beliefs(self):
        """
        Updates its beliefs
        """
        pass