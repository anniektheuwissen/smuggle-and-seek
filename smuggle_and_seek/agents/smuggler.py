import random
import numpy as np
from more_itertools import powerset

from .container import Container
from .agent import Agent

"""
Smuggler class: the smuggler agent that tries to smuggle as many drugs as possible through the containers. They
have preferences for certain containers, and they can have different levels of ToM reasoning.
"""
class Smuggler(Agent):
    def __init__(self, unique_id, model, tom_order):
        """
        Initializes the agent Smuggler
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        """
        super().__init__(unique_id, model, tom_order)
        self.distribution = []

        self.preferences = {}
        self.add_preferences()
        self.num_packages = 5

        # Define all possible distributions within those actions, and non preferences per actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_dist = []
        for i in range(1, num_cont+1):
            self.possible_dist.append(self.possible_distributions(self.num_packages,i))
        self.actions_nonpref = self.preferences_actions()

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
        print(f"preferences: {self.preferences["country"],self.preferences["cargo"]}")

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
        c_s = 1/3
        f = 1/4

        best_distributions_per_ai = [0] * len(self.possible_actions)
        for ai in range(len(self.possible_actions)):
            action_ai = self.possible_actions[ai]
            # Determine the highest phi that can be reached with this action_ai based on different distributions
            temp_phi = [0] * len(self.possible_dist[len(action_ai)-1])
            for (idx, dist) in enumerate(self.possible_dist[len(action_ai)-1]):
                # Loop over all possible actions of the opponent
                for aj in range(len(self.b0)):
                    if aj in action_ai: temp_phi[idx] += self.b0[aj] * (self.num_packages - dist[action_ai.index(aj)] - c_s*len(action_ai) - f*self.actions_nonpref[ai])
                    else: temp_phi[idx] += self.b0[aj] * (self.num_packages - c_s*len(action_ai) - f*self.actions_nonpref[ai])
            self.phi[ai] = max(temp_phi)
            self.phi[ai] = round(self.phi[ai], 4)
            best_distributions_per_ai[ai] = self.possible_dist[len(action_ai)-1][random.choice(np.where(temp_phi == max(temp_phi))[0])]
        print(f"best distributions per ai : {best_distributions_per_ai}")
        print(f"smugglers phi is : {self.phi}")
        print(f"highest index is at : {np.where(self.phi == round(max(self.phi),4))[0]}")
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi),4))[0])]

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
        print(f"hides in container {self.action}")
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for ai in self.action:
                if ai == container.unique_id:
                    container.smuggles += 1
                    container.num_packages += self.num_packages

        #PRINT:
        print("current environment:")
        print([container.num_packages for container in containers])

    def update_beliefs(self):
        """
        Updates its beliefs
        """

        # Check which actions were successful and which were not
        self.succes_actions = []; self.failed_actions = []
        containers = self.model.get_agents_of_type(Container)
        for ai in self.action:
            for container in containers:
                if container.unique_id == ai:
                    if container.num_packages == 0: self.failed_actions.append(ai)
                    else: self.succes_actions.append(ai)
        print(f"smuggler successful actions are: {self.succes_actions}, and failed actions are {self.failed_actions}")
        
        # b0
        print("smuggler is updating beliefs from ... to ...:")
        print(self.b0)
        for aj in range(len(self.b0)):
            other_actions_failed_addition = (len(self.succes_actions)/(len(containers)-len(self.succes_actions))) * (self.learning_speed/len(self.action))
            succesfull_action_addition = self.learning_speed/len(self.action)
            if aj in self.failed_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + succesfull_action_addition + other_actions_failed_addition
            if aj in self.succes_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] 
            else: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + other_actions_failed_addition
        print(self.b0)