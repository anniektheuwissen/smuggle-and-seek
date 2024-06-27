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

        self.used_by_s = 0
        self.used_by_c = 0
        self.used_succ_by_c = 0 

    def add_features(self, features):
        """
        Assigns features to the container
        """
        for i in range(len(features)):
            self.features[i] = features[i]