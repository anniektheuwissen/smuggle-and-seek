import numpy as np

"""
Strategy class: the tom0, tom1, and tom2 strategies inherit from this class
"""
class Strategy():
    def __init__(self, strategy, agent):
        """
        Initializes the strategy 
        :param strategy: the strategy
        :param agent: the agent type from which this is a strategy
        """
        self.strategy = strategy
        self.agent = agent
        
        self.print = False
    
    def softmax(self, phi, transform=1):
        """
        Calculates softmax of phi values
        :param phi: The phi values
        """
        softmax = np.exp(transform * phi) / np.sum(np.exp(transform * phi))
        return softmax