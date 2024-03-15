import mesa
import random
import numpy as np
from more_itertools import powerset

"""
Customs class: the customs agent that tries to capture as many drugs as possible from the containers. They
can have different levels of ToM reasoning.
"""
class Customs(mesa.Agent):
    def __init__(self, unique_id, model, tom_order):
        """
        Initializes the agent Customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        """
        super().__init__(unique_id, model)
        self.tom_order = tom_order
        self.points = 0
        self.action = {}

        # Define all possible actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(list, powerset(np.arange(num_cont))))[1:]

        # Initialize belief vectors and subjective value needed for tom_reasoning
        self.b0 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont-1)

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        p = 1/2

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        for ai in range(len(self.possible_actions)):
            for aj in range(len(self.b0)):
                self.phi[ai] += self.b0[aj] * (1*(aj in self.possible_actions[ai]) - p*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi[1:]),4))[0])]

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
        taking this action, and updating its beliefs
        """
        # Choose action
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        elif self.tom_order == 2: self.step_tom2()
        else: print("ERROR: Customs cannot have a theory of mind reasoning above the second order")
        
        # Take action
        print(f"checks containers {self.action}")
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for i in self.action:
                if i == container.unique_id:
                    if (container.num_packages != 0):
                        print(f"CAUGHT {container.num_packages}!!")
                        container.num_packages = 0
                    else:
                        print("wooops caught nothing")

        # Update beliefs
        # ...

"""
Smuggler class: the smuggler agent that tries to smuggle as many drugs as possible through the containers. They
have preferences for certain containers, and they can have different levels of ToM reasoning.
"""
class Smuggler(mesa.Agent):
    def __init__(self, unique_id, model, tom_order):
        """
        Initializes the agent Smuggler
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        """
        super().__init__(unique_id, model)
        self.tom_order = tom_order
        self.points = 0
        self.action = []
        self.preferences = {}
        self.num_packages = 5

        self.add_preferences()

        # Define all possible actions, all possible distributions within those actions, and non preferences per actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(list, powerset(np.arange(num_cont))))[1:]
        self.possible_dist = []
        for i in range(1, num_cont+1):
            self.possible_dist.append(self.possible_distributions(self.num_packages,i))
        self.actions_nonpref = self.preferences_actions()

        # Initialize belief vectors and subjective value needed for tom_reasoning
        self.b0 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont-1)

    def possible_distributions(self, n, m):
        """
        Determines all possible distributions of distributing n packages over m containers
        :param n The number of packages
        :param m The number of containers
        """
        if m==1: return [[n]]
        distributions = []
        for i in range(1, n-m+2):
            for subcombo in self.possible_distributions(n-i, m-1):
                distributions.append([i]+subcombo)
        return distributions
    
    def preferences_actions(self):
        """
        Determines per action how many features the used containers have that are not preferred ones
        """
        not_pref = [0] * len(self.possible_actions)
        containers = self.model.get_agents_of_type(Container)
        for (idx,action) in enumerate(self.possible_actions):
            for i in action:
                for container in containers:
                    if container.unique_id == i:
                        if container.features["cargo"] != self.preferences["cargo"]: not_pref[idx] +=1
                        if container.features["country"] != self.preferences["country"]: not_pref[idx] +=1
        return not_pref

    def add_preferences(self):
        """
        Assigns random preferences to the smuggler
        """
        self.preferences["country"] = self.random.randint(0,self.model.num_features-1)
        self.preferences["cargo"] = self.random.randint(0,self.model.num_features-1)

    def new_packages(self, x):
        """
        Gives new packages to the smuggler
        :param x: The number of packages given to the smuggler
        """
        self.num_packages = x

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        n = 3/2
        m = 1/4

        for ai in range(len(self.possible_actions)):
            action_ai = self.possible_actions[ai]
            # Determine the highest phi that can be reached with this action_ai based on different distributions
            temp_phi = [0] * len(self.possible_dist[len(action_ai)-1])
            for (idx, dist) in enumerate(self.possible_dist[len(action_ai)-1]):
                # Loop over all possible actions of the opponent
                for aj in range(len(self.b0)):
                    if aj in action_ai: temp_phi[idx] += self.b0[aj] * (self.num_packages - dist[action_ai.index(aj)] - n*len(action_ai) - m*self.actions_nonpref[ai])
                    else: temp_phi[idx] += self.b0[aj] * (self.num_packages - n*len(action_ai) - m*self.actions_nonpref[ai])
            self.phi[ai] = max(temp_phi)
            self.phi[ai] = round(self.phi[ai], 4)
        print(self.phi)
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi[1:]),4))[0])]

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        pass

    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        taking this action, and updating its beliefs
        """
        self.new_packages(5)

        # Choose action
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        else: print("ERROR: A smuggler cannot have a theory of mind reasoning above the first order")

        # Take action:
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for i in self.action:
                if i == container.unique_id:
                    container.num_packages += 10
        print(f"hides in container {self.action}")

        # Update beliefs
        # ...
        

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