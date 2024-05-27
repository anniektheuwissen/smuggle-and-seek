import numpy as np
from more_itertools import powerset

from .container import Container
from .agent import Agent

"""
Customs class: the customs agent that tries to capture as many drugs as possible from the containers. They
can have different levels of ToM reasoning.
"""
class Customs(Agent):
    def __init__(self, unique_id, model, tom_order, learning_speed):
        """
        Initializes the agent Customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed: The speed at which the agent learns
        """
        super().__init__(unique_id, model, tom_order, learning_speed)
        self.container_costs = 4

        self.num_checks = 0
        self.successful_checks = 0
        self.catched_packages = 0

        self.expected_amount_catch = 1
        self.expected_preferences = {}
        self.expected_preferences["country"] = self.random.randint(0,1)
        self.expected_preferences["cargo"] = self.random.randint(0,1)
    
    def reward_function(self, c, aj):
        """
        Returns the reward based on the reward function of customs
        :param c: The container that customs use
        :param aj: The action of the smuggler
        """
        c_c = self.container_costs
        return (2*self.expected_amount_catch*(c in self.possible_actions[aj]) - c_c*len(self.possible_actions[aj]))
    
    def simulation_reward_function(self, c, aj):
        """
        Returns the reward based on the simulated reward function of the smuggler
        :param c: The container that customs use
        :param aj: The action of the smuggler
        """
        non_pref = (self.model.get_agents_of_type(Container)[aj].features["cargo"] != self.expected_preferences["cargo"]) + (self.model.get_agents_of_type(Container)[aj].features["country"] != self.expected_preferences["country"])
        return (-1*(aj == c) +1*(aj != c) - non_pref)

    def calculate_phi(self, actions, beliefs, reward_function):
        """
        Calculates the subjective value phi of all possible actions
        :param actions: The possible actions for which to calculate a phi value
        :param beliefs: The beliefs based on which the phi values have to be calculated
        :param reward_function: The reward function with which the phi values have to be calculated
        """
        phi = np.zeros(len(actions))
        
        for ai in range(len(actions)):
            for c in range(len(beliefs)):
                if reward_function == "normal": reward = self.reward_function(c, ai)
                elif reward_function == "simulation": reward = self.simulation_reward_function(c, ai)
                phi[ai] += beliefs[c] * reward

        if reward_function == "normal": self.phi = phi
        elif reward_function == "simulation": self.simulation_phi = phi

    def choose_action_softmax(self):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        """
        softmax_phi = np.exp(self.phi) / np.sum(np.exp(self.phi))
        if self.model.print: print(f"customs softmax of phi is : {softmax_phi}")
        action_indexes = [i for i in range(0,len(self.possible_actions))]
        index_action = np.random.choice(action_indexes, 1, p=softmax_phi)[0]
        self.action = self.possible_actions[index_action]

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        self.calculate_phi(self.possible_actions, self.b0, "normal")
        if self.model.print: print(f"custom's phi is : {self.phi}")
        self.choose_action_softmax()

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        # Make prediction about behavior of opponent
        self.calculate_phi(self.b1, self.b1, "simulation")
        if self.model.print: print(f"custom's simulation phi is : {self.simulation_phi}")
        self.prediction_a1 = np.exp(self.simulation_phi) / np.sum(np.exp(self.simulation_phi))     
        if self.model.print: print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        W = self.merge_prediction()
        if self.model.print: print(f"W is : {W}")

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        self.calculate_phi(self.possible_actions, W, "normal")
        if self.model.print: print(f"custom's phi is : {self.phi}")
        self.choose_action_softmax()

    def step_tom2(self):
        """
        Chooses an action associated with second-order theory of mind reasoning
        """
        if self.model.print: print("I am a second order ToM customs")
        pass

    def take_action(self):
        """
        Performs action and find out succes/failure of action
        """
        if self.model.print: print(f"checks containers {self.action}")
        self.failed_actions = []; self.succes_actions = []
        for ai in self.action:
            container = self.model.get_agents_of_type(Container)[ai]
            container.used_by_c += 1
            if (container.num_packages != 0):
                if self.model.print: print(f"caught {container.num_packages} packages!!")
                self.catched_packages += container.num_packages
                self.succes_actions.append(ai)
                container.num_packages = 0
                container.used_succ_by_c += 1
            else:
                if self.model.print: print("wooops caught nothing")
                self.failed_actions.append(ai)
        if self.model.print: print(f"customs succesfull actions are: {self.succes_actions}, and failed actions are: {self.failed_actions}")
        self.num_checks += len(self.action); self.successful_checks += len(self.succes_actions)
        
    def update_expected_amount_catch(self):
        """
        Updates the expected amount of packages of one catch
        """
        if self.successful_checks > 0:
            self.expected_amount_catch = self.catched_packages / self.successful_checks
            if self.model.print: print(f"expected amount catch is: {self.expected_amount_catch}")

    def update_b0(self, f, n):
        """
        Updates b0
        """
        if self.model.print: print("customs are updating beliefs b0 from ... to ...:")
        if self.model.print: print(self.b0)
        if len(self.succes_actions) > 0:
            a = (self.learning_speed/f)/len(self.succes_actions)
            for c in range(len(self.b0)):
                cf_succ = 0
                for c_star in self.succes_actions: cf_succ += self.common_features(c, c_star)
                self.b0[c] = (1 - self.learning_speed) * self.b0[c] + a * cf_succ
        elif len(self.failed_actions) > 0:
            b = self.learning_speed/2/(n-len(self.failed_actions))
            for c in range(len(self.b0)):
                self.b0[c] = (1 - self.learning_speed/2) * self.b0[c] + b * (c not in self.failed_actions)
        if self.model.print: print(self.b0)

    def update_expected_preferences(self):
        """
        Updates the expected preferences of the smuggler
        """
        containers = self.model.get_agents_of_type(Container)
        checked_country0 = 0; checked_country1 = 0; checked_cargo0 = 0; checked_cargo1 = 0
        for container in containers:
            if container.features["country"] == 0: checked_country0 += container.used_succ_by_c
            if container.features["country"] == 1: checked_country1 += container.used_succ_by_c
            if container.features["cargo"] == 0: checked_cargo0 += container.used_succ_by_c
            if container.features["cargo"] == 1: checked_cargo1 += container.used_succ_by_c
        if checked_country0 > checked_country1: self.expected_preferences["country"] = 0
        else: self.expected_preferences["country"] = 1
        if checked_cargo0 > checked_cargo1: self.expected_preferences["cargo"] = 0
        else: self.expected_preferences["cargo"] = 1
        if self.model.print: print(f"expected preferences are: {self.expected_preferences}")

    def update_b1(self, f, n):
        """
        Updates b1
        """
        if self.model.print: print("customs are updating beliefs b1 from ... to ...:")
        if self.model.print: print(self.b1)
        if len(self.succes_actions) > 0:
            a = (self.learning_speed/f)/len(self.succes_actions)
            for c in range(len(self.b1)):
                cf_succ = 0
                for c_star in self.succes_actions: cf_succ += self.common_features(c, c_star)
                self.b1[c] = (1 - self.learning_speed) * self.b1[c] + a * cf_succ
        elif len(self.failed_actions) > 0:
            b = (self.learning_speed/(2*n))/len(self.failed_actions)
            for c in range(len(self.b1)):
                self.b1[c] = (1 - self.learning_speed/(2*n)) * self.b1[c] + b * (c in self.failed_actions)
        if self.model.print: print(self.b1)

    def update_c1(self):
        """
        Updates c1
        """
        if self.model.print: print("customs are updating c1 from ... to ...:")
        if self.model.print: print(self.c1)
        for a in self.action:
            action_index = self.possible_actions.index(a)
            prediction = self.prediction_a1[action_index]
            if prediction < 0.25:
                update = 0.25 - prediction
                if a in self.failed_actions: self.c1 = (1 - update) * self.c1 + update;
                if a in self.succes_actions: self.c1 = (1 - update) * self.c1;
            if prediction > 0.25:
                update = prediction - 0.25
                if a in self.failed_actions: self.c1 = (1 - update) * self.c1;
                if a in self.succes_actions: self.c1 = (1 - update) * self.c1 + update;
        if self.model.print: print(self.c1)

    def update_beliefs(self):
        """
        Updates its beliefs, confidence and expectations
        """
        f = self.model.i_per_feat * self.model.num_features
        n = self.model.i_per_feat ** self.model.num_features

        self.update_expected_amount_catch()
        self.update_b0(f, n)

        if self.tom_order > 0:
            self.update_expected_preferences()
            self.update_b1(f, n)
            self.update_c1()
