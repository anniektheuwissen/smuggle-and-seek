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
    def __init__(self, unique_id, model, tom_order, learning_speed1, learning_speed2, packages):
        """
        Initializes the agent Smuggler
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed1: The speed at which the agent learns in most situations
        :param learning_speed2: The speed at which the agent learns in less informative situations
        """
        super().__init__(unique_id, model, tom_order, learning_speed1, learning_speed2)
        self.container_costs = 3
        self.feature_costs = 1

        self.distribution = []
        self.preferences = {}
        self.add_preferences()
        self.num_packages = packages

        self.num_smuggles = 0
        self.successful_smuggles = 0
        self.successful_smuggled_packages = 0
        self.failed_smuggles = 0
        self.failed_packages = 0
        self.nonpref_used = 0 
        self.average_amount_catch = 5

        # Redefine possible actions and phi
        self.possible_actions = list(filter(lambda x: len(x) <= packages, self.possible_actions))
        self.phi = self.phi[:len(self.possible_actions)]

        # Define all possible distributions within the actions, and non preferences per actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_dist = []
        for i in range(1, num_cont+1):
            self.possible_dist.append(self.possible_distributions(self.model.packages_per_day,i))
        self.actions_nonpref = self.preferences_actions()
        self.best_distributions_per_ai = [0] * len(self.possible_actions)

    def add_preferences(self):
        """
        Assigns random preferences to the smuggler
        """
        self.preferences["country"] = self.random.randint(0,self.model.num_features-1)
        self.preferences["cargo"] = self.random.randint(0,self.model.num_features-1)
        if self.model.print: print(f"preferences: {self.preferences["country"],self.preferences["cargo"]}")

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
        for (idx,action) in enumerate(self.possible_actions):
            for i in action:
                if self.model.get_agents_of_type(Container)[i].features["cargo"] != self.preferences["cargo"]: not_pref[idx] +=1
                if self.model.get_agents_of_type(Container)[i].features["country"] != self.preferences["country"]: not_pref[idx] +=1
        return not_pref
    
    def calculate_phi(self, beliefs):
        """
        Calculates the subjective value phi of all possible actions and all their distributions
        :param beliefs: The beliefs based on which the phi values have to be calculated
        """
        c_s = self.container_costs; f = self.feature_costs
        self.best_distributions_per_ai = [0] * len(self.possible_actions)

        for ai in range(len(self.possible_actions)):
            action_ai = self.possible_actions[ai]
            # Determine the highest phi that can be reached with this action_ai based on different distributions
            temp_phi = [0] * len(self.possible_dist[len(action_ai)-1])
            for (idx, dist) in enumerate(self.possible_dist[len(action_ai)-1]):
                # Loop over all possible actions of the opponent
                for aj in range(len(beliefs)):
                    if aj in action_ai: temp_phi[idx] += beliefs[aj] * (2*(self.num_packages - dist[action_ai.index(aj)]) - c_s*len(action_ai) - f*self.actions_nonpref[ai])
                    else: temp_phi[idx] += beliefs[aj] * (2*self.num_packages - c_s*len(action_ai) - f*self.actions_nonpref[ai])
            if self.model.print: print(f"{ai}, {action_ai}: {temp_phi}")
            self.phi[ai] = max(temp_phi)
            self.phi[ai] = round(self.phi[ai], 4)
            self.best_distributions_per_ai[ai] = self.possible_dist[len(action_ai)-1][random.choice(np.where(temp_phi == max(temp_phi))[0])]

    def calculate_simulation_phi(self):
        """
        Calculates the subjective value phi of police by using beliefs b1 and a simulated reward function
        """
        self.simulation_phi = np.zeros(len(self.b0))
        for c in range(len(self.b1)):
            for c_star in range(len(self.b1)):
                self.simulation_phi[c] += self.b1[c_star] * (1*self.average_amount_catch*(c == c_star) -1*self.average_amount_catch*(c != c_star))
    
    def choose_action_softmax(self):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        """
        softmax_phi = np.exp(self.phi) / np.sum(np.exp(self.phi))
        if self.model.print: print(f"smugglers softmax of phi is : {softmax_phi}")
        action_indexes = [i for i in range(0,len(self.possible_actions))]
        index_action = np.random.choice(action_indexes, 1, p=softmax_phi)[0]
        self.action = self.possible_actions[index_action]
        self.distribution = self.best_distributions_per_ai[index_action]

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        self.calculate_phi(self.b0)
        if self.model.print: print(f"best distributions per ai : {self.best_distributions_per_ai}")
        if self.model.print: print(f"smugglers phi is : {self.phi}")
        self.choose_action_softmax()

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        # Make prediction about behavior of opponent
        self.calculate_simulation_phi()
        if self.model.print: print(f"smugglers simulation phi is : {self.simulation_phi}")   
        self.prediction_a1 = np.exp(self.simulation_phi) / np.sum(np.exp(self.simulation_phi))       
        if self.model.print: print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        W = self.merge_prediction(self.prediction_a1, self.b0, self.c1)
        if self.model.print: print(f"W is : {W}")

        # Make decision
        # Calculate the subjective value phi for each action, and choose the action with the highest.
        self.calculate_phi(W)
        if self.model.print: print(f"best distributions per ai : {self.best_distributions_per_ai}")
        if self.model.print: print(f"smugglers phi is : {self.phi}")
        self.choose_action_softmax()

    def take_action(self):
        """
        Performs action
        """
        if self.model.print: print(f"hides in container {self.action} with distribution {self.distribution}")
        for (idx, ai) in enumerate(self.action):
            self.model.get_agents_of_type(Container)[ai].used_by_s += 1
            self.model.get_agents_of_type(Container)[ai].num_packages += self.distribution[idx]
            self.num_smuggles += 1
            self.nonpref_used += self.actions_nonpref[ai]

    def check_result_actions(self):
        """
        Checks the results of the taken action, i.e. which actions were successful and which were not
        """
        self.succes_actions = []; self.failed_actions = []
        containers = self.model.get_agents_of_type(Container)
        for ai in self.action:
            for container in containers:
                if container.unique_id == ai:
                    if container.num_packages == 0: self.failed_actions.append(ai); self.failed_packages += self.distribution[self.action.index(ai)]; self.failed_smuggles += 1
                    else: self.succes_actions.append(ai); self.successful_smuggled_packages += container.num_packages; self.successful_smuggles += 1
        if self.model.print: print(f"smuggler successful actions are: {self.succes_actions}, and failed actions are {self.failed_actions}")
    
    def update_average_amount_per_catch(self):
        if (self.num_smuggles - self.successful_smuggles) > 0:
            self.average_amount_catch = self.failed_packages / self.failed_smuggles
        
    def update_b0(self, f, n):
        """
        Updates b0
        """
        if self.model.print: print("smuggler is updating beliefs b0 from ... to ...:")
        if self.model.print: print(self.b0)
        if len(self.failed_actions) > 0:
            for c in range(len(self.b0)):
                if c in self.failed_actions:
                    self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.failed_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.failed_actions)
                    self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + similarity * self.learning_speed1
        elif len(self.succes_actions) > 0:
            for c in range(len(self.b0)):
                self.b0[c] = (1 - self.learning_speed2) * self.b0[c] + (c not in self.succes_actions) * self.learning_speed2
        if self.model.print: print(self.b0)

    def update_b1(self, f, n):
        """
        Updates b1
        """
        if self.model.print: print("smuggler is updating beliefs b1 from ... to ...:")
        if self.model.print: print(self.b1)
        if len(self.failed_actions) > 0:
            for c in range(len(self.b1)):
                if c in self.failed_actions:
                    self.b1[c] = (1 - self.learning_speed1) * self.b1[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.failed_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.failed_actions)
                    self.b1[c] = (1 - self.learning_speed1) * self.b1[c] + similarity * self.learning_speed1
        elif len(self.succes_actions) > 0:
            for c in range(len(self.b1)):
                self.b1[c] = (1 - self.learning_speed2) * self.b1[c] + (c in self.succes_actions) * self.learning_speed2
        if self.model.print: print(self.b1)
    
    def update_c1(self):
        """
        Updates c1
        """
        if self.model.print: print("smuggler is updating c1 from ... to ...:")
        if self.model.print: print(self.c1)
        for a in self.action:
            action_index = self.possible_actions.index(a)
            prediction = self.prediction_a1[action_index]
            if prediction < 0.25:
                update = 0.25 - prediction
                if a in self.succes_actions: self.c1 = (1 - update) * self.c1 + update;
                if a in self.failed_actions: self.c1 = (1 - update) * self.c1;
            if prediction > 0.25:
                update = prediction - 0.25
                if a in self.succes_actions: self.c1 = (1 - update) * self.c1;
                if a in self.failed_actions: self.c1 = (1 - update) * self.c1 + update;
        if self.model.print: print(self.c1)

    def update_beliefs(self):
        """
        Updates its beliefs and confidence
        """
        f = self.model.i_per_feat * self.model.num_features
        n = self.model.i_per_feat ** self.model.num_features

        self.check_result_actions()
        self.update_average_amount_per_catch()
        self.update_b0(f, n)

        if self.tom_order > 0:
            self.update_b1(f, n)
            self.update_c1()