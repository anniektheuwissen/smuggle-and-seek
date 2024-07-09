from more_itertools import powerset


"""
Strategy class: the tom0, tom1, and tom2 strategies inherit from this class
"""
class Strategy():
    def __init__(self, strategy):
        """
        Initializes the strategy 
        :param strategy: the strategy
        """
        self.strategy = strategy
        
        self.print = False

    
