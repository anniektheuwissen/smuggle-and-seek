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
    def __init__(self, unique_id, model, tom_order, learning_speed, exploration_exploitation):
        """
        Initializes the agent Smuggler
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed: The speed at which the agent learns
        """
        super().__init__(unique_id, model, tom_order, learning_speed, exploration_exploitation)
        self.container_costs = 1/4
        self.feature_costs = 1/4

        self.distribution = []
        self.preferences = {}
        self.add_preferences()
        self.num_packages = 0

        self.successful_smuggled_packages = 0

        # Define all possible distributions within the actions, and non preferences per actions
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_dist = []
        for i in range(1, num_cont+1):
            self.possible_dist.append(self.possible_distributions(self.model.packages_per_day,i))
        self.actions_nonpref = self.preferences_actions()


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

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        c_s = self.container_costs
        f = self.feature_costs

        best_distributions_per_ai = [0] * len(self.possible_actions)
        for ai in range(len(self.possible_actions)):
            action_ai = self.possible_actions[ai]
            # Determine the highest phi that can be reached with this action_ai based on different distributions
            temp_phi = [0] * len(self.possible_dist[len(action_ai)-1])
            for (idx, dist) in enumerate(self.possible_dist[len(action_ai)-1]):
                # Loop over all possible actions of the opponent
                for aj in range(len(self.b0)):
                    if aj in action_ai: temp_phi[idx] += self.b0[aj] * (self.num_packages - 2*dist[action_ai.index(aj)] - c_s*len(action_ai) - f*self.actions_nonpref[ai])
                    else: temp_phi[idx] += self.b0[aj] * (self.num_packages - c_s*len(action_ai) - f*self.actions_nonpref[ai])
            print(f"{ai}, {action_ai}: {temp_phi}")
            self.phi[ai] = max(temp_phi)
            self.phi[ai] = round(self.phi[ai], 4)
            best_distributions_per_ai[ai] = self.possible_dist[len(action_ai)-1][random.choice(np.where(temp_phi == max(temp_phi))[0])]
        print(f"best distributions per ai : {best_distributions_per_ai}")
        print(f"smugglers phi is : {self.phi}")
        print(round(max(self.phi),2))
        print(f"highest index is at : {np.where(self.phi == round(max(self.phi),2))[0]}")
        index_action = random.choice(np.where(self.phi == round(max(self.phi),4))[0])
        self.action = self.possible_actions[index_action]
        self.distribution = best_distributions_per_ai[index_action]

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        print("I am a first order ToM smuggler")

        c_s = self.container_costs
        f = self.feature_costs

        # Make prediction about behavior of opponent
        simulation_phi = np.zeros(len(self.b1))
        for c in range(len(self.b1)):
            for c_star in range(len(self.b1)):
                simulation_phi[c] += self.b1[c_star] * (1*(c == c_star) -1*(c != c_star))
        print(f"smugglers simulation phi is : {simulation_phi}")
        smallest = simulation_phi[0]
        for i in simulation_phi:
            if i < smallest:
                smallest = i
        if smallest < 0:
            for i in range(len(simulation_phi)):
                simulation_phi[i] -= smallest
            print(f"updated to.... {simulation_phi}")          
        for i in range(len(self.prediction_a1)):
            if sum(simulation_phi) == 0: self.prediction_a1[i] = 0
            else: self.prediction_a1[i] = round(simulation_phi[i] /sum(simulation_phi),2)
        print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        W = np.zeros(len(self.b1))
        for c in range(len(self.b1)):
            W[c] = self.c1 * self.prediction_a1[c] + (1-self.c1) * self.b0[c]
        print(f"W is : {W}")

        # Make decision
        # Calculate the subjective value phi for each action, and choose the action with the highest.
        best_distributions_per_ai = [0] * len(self.possible_actions)
        for ai in range(len(self.possible_actions)):
            action_ai = self.possible_actions[ai]
            # Determine the highest phi that can be reached with this action_ai based on different distributions
            temp_phi = [0] * len(self.possible_dist[len(action_ai)-1])
            for (idx, dist) in enumerate(self.possible_dist[len(action_ai)-1]):
                # Loop over all possible actions of the opponent
                for c in range(len(W)):
                    if c in action_ai: temp_phi[idx] += W[c] * (self.num_packages - 2*dist[action_ai.index(c)] - c_s*len(action_ai) - f*self.actions_nonpref[ai])
                    else: temp_phi[idx] += W[c] * (self.num_packages - c_s*len(action_ai) - f*self.actions_nonpref[ai])
            print(f"{ai}, {action_ai}: {temp_phi}")
            self.phi[ai] = max(temp_phi)
            self.phi[ai] = round(self.phi[ai], 4)
            best_distributions_per_ai[ai] = self.possible_dist[len(action_ai)-1][random.choice(np.where(temp_phi == max(temp_phi))[0])]
        print(f"best distributions per ai : {best_distributions_per_ai}")
        print(f"smugglers phi is : {self.phi}")
        print(max(self.phi))
        print(f"highest index is at : {np.where(self.phi == max(self.phi))[0]}")
        index_action = random.choice(np.where(self.phi == max(self.phi))[0])
        self.action = self.possible_actions[index_action]
        self.distribution = best_distributions_per_ai[index_action]

    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        taking this action, and updating its beliefs
        """
        # Reset phi and receive new packages
        for i in range(len(self.phi)): self.phi[i] = 0
        self.new_packages(self.model.packages_per_day)

        # Choose action based on the order of tom reasoning
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        else: print("ERROR: A smuggler cannot have a theory of mind reasoning above the first order")

        # Random checkups:
        if self.exploration_exploitation:
            if (random.randint(0,99) < 1):
                self.action = random.choice(self.possible_actions)
                self.distribution = random.choice(self.possible_dist[len(self.action)-1])

        # Take action:
        print(f"hides in container {self.action} with distribution {self.distribution}")
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for (idx,ai) in enumerate(self.action):
                if ai == container.unique_id:
                    print(f"{idx}, {ai}: {self.distribution[idx]}")
                    container.used_s += 1
                    container.num_packages += self.distribution[idx]

        #PRINT:
        print("current environment:")
        print([container.num_packages for container in containers])

    def update_beliefs(self):
        """
        Updates its beliefs
        """
        if self.action == []:
            pass
        else: 
            # Check which actions were successful and which were not
            self.succes_actions = []; self.failed_actions = []
            containers = self.model.get_agents_of_type(Container)
            for ai in self.action:
                for container in containers:
                    if container.unique_id == ai:
                        if container.num_packages == 0: self.failed_actions.append(ai)
                        else: self.succes_actions.append(ai); self.successful_smuggled_packages += container.num_packages
            print(f"smuggler successful actions are: {self.succes_actions}, and failed actions are {self.failed_actions}")
            
            # Update b0
            print("smuggler is updating beliefs b0 from ... to ...:")
            print(self.b0)
            for aj in range(len(self.b0)):
                other_actions_succeeded_addition = (len(self.succes_actions)/(len(containers)-len(self.succes_actions))) * (self.learning_speed/len(self.action))
                failed_action_addition = self.learning_speed/len(self.action)
                if aj in self.failed_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + failed_action_addition + other_actions_succeeded_addition
                elif aj in self.succes_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] 
                else: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + other_actions_succeeded_addition
            print(self.b0)

            if self.tom_order > 0:
                # Update b1
                print("smuggler is updating beliefs b1 from ... to ...:")
                print(self.b1)
                if len(self.failed_actions) > 0:
                    for aj in range(len(self.b1)):
                        failed_action_addition = self.learning_speed/len(self.failed_actions)
                        if aj in self.failed_actions: self.b1[aj] = (1 - self.learning_speed) * self.b1[aj] + failed_action_addition
                        else: self.b1[aj] = (1 - self.learning_speed) * self.b1[aj]
                elif len(self.succes_actions) > 0:
                    for aj in range(len(self.b1)):
                        succesfull_action_addition = (self.learning_speed/3)/len(self.succes_actions)
                        if aj in self.succes_actions: self.b1[aj] = (1 - self.learning_speed/3) * self.b1[aj] + succesfull_action_addition
                        else: self.b1[aj] = (1 - self.learning_speed/3) * self.b1[aj]
                else: print("stays the same...")
                print(self.b1)

                # Update c1
                update_speed = 0.2

                print("smuggler is updating c1 from ... to ...:")
                print(self.c1)
                max_indexes_prediction = np.where(self.prediction_a1 == max(self.prediction_a1))[0]
                print(f"max indexes prediction to update from are: {max_indexes_prediction}")
                for c in max_indexes_prediction:
                    if c in self.succes_actions:
                        self.c1 = (1 - update_speed) * self.c1
                    elif c in self.failed_actions:
                        self.c1 = (1 - update_speed) * self.c1 + update_speed
                if (not any(c in self.action for c in max_indexes_prediction) and len(self.failed_actions) == 0):
                    self.c1 = (1 - update_speed/3) * self.c1 + update_speed/3
                print(self.c1)