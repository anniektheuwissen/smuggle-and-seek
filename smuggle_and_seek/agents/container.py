import mesa

"""
Container class: the container agent that is used to hide drugs in. They have features.
"""
class Container(mesa.Agent):
    def __init__(self, unique_id, model):
        """
        Initializes the agent Container
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        """
        super().__init__(unique_id, model)
        self.features = {}
        self.num_packages = 0
        self.smuggles = 0
        self.checks = 0

    def add_features(self, x, y):
        """
        Assigns features to the container
        """
        self.features["country"] = x
        self.features["cargo"] = y

    def country_feature(self):
        """
        Returns the feature 'country' of the container
        """
        return self.features["country"]

    def cargo_feature(self):
        """
        Returns the feature 'cargo' of the container
        """
        return self.features["cargo"]