import mesa
import numpy as np

from .container import Container
from .strategies.tom0 import Tom0
from .strategies.tom1 import Tom1
from .strategies.tom2 import Tom2

"""
SmuggleAndSeekAgent class: the customs agent and smuggler agent inherit from this class
"""
class SmuggleAndSeekAgent(mesa.Agent):
    def __init__(self, unique_id, model, tom_order, learning_speed1, learning_speed2, agenttype):
        """
        Initializes the agent 
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed1: The speed at which the agent learns in most situations
        :param learning_speed2: The speed at which the agent learns in less informative situations
        :param agenttype: The type of the agent, i.e. "smuggler" or "customs"
        """
        super().__init__(unique_id, model)

        match tom_order:
            case 0: self.strategy = Tom0(agenttype)
            case 1: self.strategy = Tom1(agenttype)
            case 2: self.strategy = Tom2(agenttype)

        self.points = 0
        self.points_queue = [0] * 10
        self.action = []
        self.failed_actions = []
        self.succes_actions = []

        self.learning_speed1 = learning_speed1
        self.learning_speed2 = learning_speed2

        num_cont = len(self.model.get_agents_of_type(Container))
        # Initialize belief vectors and confidence needed for tom_reasoning
        self.b0 = np.array([1/num_cont] * num_cont)
        self.b1 = np.array([1/num_cont] * num_cont)
        self.b2 = np.array([1/num_cont] * num_cont)
        self.conf1 = 0
        self.conf2 = 0

    def similarity(self, c1, c2):
        """
        Calculates the similarity between two container types
        :param c1: Container type 1
        :param c2: Container type 2
        """
        container_c1 = self.model.get_agents_of_type(Container)[c1]; container_c2 = self.model.get_agents_of_type(Container)[c2]
        similarity = 0
        for i in range(len(container_c1.features)):
            similarity += (container_c1.features[i] == container_c2.features[i])
        similarity /= len(container_c1.features)
        return similarity