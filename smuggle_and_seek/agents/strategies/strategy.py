
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

    
