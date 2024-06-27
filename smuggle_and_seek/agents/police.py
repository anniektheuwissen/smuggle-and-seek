import numpy as np
from more_itertools import powerset

from .container import Container
from .agent import Agent

"""
police class: the police agent that tries to capture as many drugs as possible from the containers. They
can have different levels of ToM reasoning.
"""
class Police(Agent):
    def __init__(self, unique_id, model, tom_order, learning_speed1, learning_speed2):
        """
        Initializes the agent police
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed1: The speed at which the agent learns in most situations
        :param learning_speed2: The speed at which the agent learns in less informative situations
        """
        super().__init__(unique_id, model, tom_order, learning_speed1, learning_speed2)
        self.container_costs = 4

        self.num_checks = 0
        self.successful_checks = 0
        self.catched_packages = 0

        self.expected_amount_catch = 1
        self.expected_preferences = {}
        self.expected_preferences[0] = self.random.randint(0,1)
        self.expected_preferences[1] = self.random.randint(0,1)

        num_cont = len(self.model.get_agents_of_type(Container))
        self.prediction_a2 = np.zeros(len(self.prediction_a1))
        self.b2 = np.array([1/num_cont] * num_cont)
        self.c1_sim = 0.8
        self.c2 = 0
    
    def reward_function(self, c, aj):
        """
        Returns the reward based on the reward function of police
        :param c: The container that police use
        :param aj: The action of the smuggler
        """
        c_c = self.container_costs
        return (2*self.expected_amount_catch*(c in self.possible_actions[aj]) - c_c*len(self.possible_actions[aj]))
    
    def simulation_reward_function(self, c, aj):
        """
        Returns the reward based on the simulated reward function of the smuggler
        :param c: The container that police use
        :param aj: The action of the smuggler
        """
        non_pref = (self.model.get_agents_of_type(Container)[aj].features[0] != self.expected_preferences[0]) + (self.model.get_agents_of_type(Container)[aj].features[1] != self.expected_preferences[1])
        return (-1*self.expected_amount_catch*(aj == c) +1*self.expected_amount_catch*(aj != c) - non_pref)

    def smugglers_simulation_reward_function(self, c, ai):
        return (1*self.expected_amount_catch*(ai == c) -1*self.expected_amount_catch*(ai != c))

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
                elif reward_function == "simulation2": reward = self.smugglers_simulation_reward_function(c, ai)
                phi[ai] += beliefs[c] * reward

        if reward_function == "normal": self.phi = phi
        elif reward_function == "simulation" or reward_function == "simulation2": self.simulation_phi = phi

    def choose_action_softmax(self):
        """
        Chooses an action to play based on the softmax over the subjective value phi
        """
        softmax_phi = np.exp(self.phi) / np.sum(np.exp(self.phi))
        if self.model.print: print(f"police softmax of phi is : {softmax_phi}")
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
        W = self.merge_prediction(self.prediction_a1, self.b0, self.c1)
        if self.model.print: print(f"W is : {W}")

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        self.calculate_phi(self.possible_actions, W, "normal")
        if self.model.print: print(f"custom's phi is : {self.phi}")
        self.choose_action_softmax()

    def step_tom2(self):
        """
        Chooses an action associated with second-order theory of mind reasoning
        """
        if self.model.print:
            print("")
            print("police' TOM2 state:")
            print(f"b0 is : {self.b0}")
            print(f"b1 is : {self.b1}")
            print(f"b2 is : {self.b2}")
            print(f"bp is : {self.expected_preferences}")
            print(f"c1 is : {self.c1}")
            print(f"c2 is : {self.c2}")
        # Make prediction about behavior of opponent
        #   First make prediction about prediction that tom1 smuggler would make about behavior police
        self.calculate_phi(self.b2, self.b2, "simulation2")
        if self.model.print: print(f"custom's prediction of smuggler's simulation phi is : {self.simulation_phi}")
        self.prediction_a1 = np.exp(self.simulation_phi) / np.sum(np.exp(self.simulation_phi)) 
        if self.model.print: print(f"custom's prediction of smuggler's prediction a1 is : {self.prediction_a1}")
        #   Merge this prediction that tom1 smuggler would make with b1 (represents b0 of smuggler) 
        W = self.merge_prediction(self.prediction_a1, self.b1, self.c1_sim)
        if self.model.print: print(f"W is : {W}")

        #   Then use this prediction of integrated beliefs of the opponent to predict the behavior of the opponent
        self.calculate_phi(W, W, "simulation")
        if self.model.print: print(f"custom's simulation phi is : {self.simulation_phi}")
        self.prediction_a2 = np.exp(self.simulation_phi) / np.sum(np.exp(self.simulation_phi))     
        if self.model.print: print(f"prediction a2 is : {self.prediction_a2}")

        # Merge prediction with integrated beliefs of first-order prediction and zero-order beliefs
        # Make first-order prediction about behavior of opponent
        if self.model.print: print(f"police are calculating first-order prediction.....")
        self.calculate_phi(self.b1, self.b1, "simulation")
        if self.model.print: print(f"custom's simulation phi is : {self.simulation_phi}")
        self.prediction_a1 = np.exp(self.simulation_phi) / np.sum(np.exp(self.simulation_phi))     
        if self.model.print: print(f"prediction a1 is : {self.prediction_a1}")
        # Merge first-order prediction with zero-order belief
        W = self.merge_prediction(self.prediction_a1, self.b0, self.c1)
        if self.model.print: print(f"W is : {W}")
        # Merge second-order prediction with integrated belief
        W2 = self.merge_prediction(self.prediction_a2, W, self.c2)
        if self.model.print: print(f"W2 is : {W2}")

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        self.calculate_phi(self.possible_actions, W2, "normal")
        if self.model.print: print(f"custom's phi is : {self.phi}")
        self.choose_action_softmax()


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
        if self.model.print: print(f"police succesfull actions are: {self.succes_actions}, and failed actions are: {self.failed_actions}")
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
        if self.model.print: print("police are updating beliefs b0 from ... to ...:")
        if self.model.print: print(self.b0)
        if len(self.succes_actions) > 0:
            for c in range(len(self.b0)):
                if c in self.succes_actions:
                    self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.succes_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.succes_actions)
                    self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + similarity * self.learning_speed1
        elif len(self.failed_actions) > 0:
            for c in range(len(self.b0)):
                self.b0[c] = (1 - self.learning_speed2) * self.b0[c] + (c not in self.failed_actions) * self.learning_speed2
        if self.model.print: print(self.b0)

    def update_expected_preferences(self):
        """
        Updates the expected preferences of the smuggler
        """
        containers = self.model.get_agents_of_type(Container)
        checked_country0 = 0; checked_country1 = 0; checked_cargo0 = 0; checked_cargo1 = 0
        for container in containers:
            if container.features[0] == 0: checked_country0 += container.used_succ_by_c
            if container.features[0] == 1: checked_country1 += container.used_succ_by_c
            if container.features[1] == 0: checked_cargo0 += container.used_succ_by_c
            if container.features[1] == 1: checked_cargo1 += container.used_succ_by_c
        if checked_country0 > checked_country1: self.expected_preferences[0] = 0
        else: self.expected_preferences[0] = 1
        if checked_cargo0 > checked_cargo1: self.expected_preferences[1] = 0
        else: self.expected_preferences[1] = 1
        if self.model.print: print(f"expected preferences are: {self.expected_preferences}")

    def update_b1(self, f, n):
        """
        Updates b1
        """
        if self.model.print: print("police are updating beliefs b1 from ... to ...:")
        if self.model.print: print(self.b1)
        if len(self.succes_actions) > 0:
            for c in range(len(self.b1)):
                if c in self.succes_actions:
                    self.b1[c] = (1 - self.learning_speed1) * self.b1[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.succes_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.succes_actions)
                    self.b1[c] = (1 - self.learning_speed1) * self.b1[c] + similarity * self.learning_speed1
        elif len(self.failed_actions) > 0:
            for c in range(len(self.b1)):
                self.b1[c] = (1 - self.learning_speed2) * self.b1[c] + (c in self.failed_actions) * self.learning_speed2
        if self.model.print: print(self.b1)
    
    def update_b2(self, f, n):
        """
        Updates b2
        """
        if self.model.print: print("police are updating beliefs b2 from ... to ...:")
        if self.model.print: print(self.b2)
        if len(self.succes_actions) > 0:
            a = (self.learning_speed1/f)/len(self.succes_actions)
            for c in range(len(self.b2)):
                cf_succ = 0
                for c_star in self.succes_actions: cf_succ += self.common_features(c, c_star)
                self.b2[c] = (1 - self.learning_speed1) * self.b2[c] + a * cf_succ
        elif len(self.failed_actions) > 0:
            b = (self.learning_speed1/(2*n))/(n - len(self.failed_actions))
            for c in range(len(self.b1)):
                self.b2[c] = (1 - self.learning_speed1/(2*n)) * self.b2[c] + b * (c not in self.failed_actions)
        if self.model.print: print(self.b2)

    def update_confidence(self, confidence):
        """
        Updates confidence (c1 or c2)
        """
        if self.model.print: print("police are updating confidence from ... to ...:")
        if self.model.print: print(confidence)
        for a in self.action:
            action_index = self.possible_actions.index(a)
            if confidence == self.c1: prediction = self.prediction_a1[action_index]
            elif confidence == self.c2: prediction = self.prediction_a2[action_index]
            if prediction < 0.25:
                update = 0.25 - prediction
                if a in self.failed_actions: confidence = (1 - update) * confidence + update;
                if a in self.succes_actions: confidence = (1 - update) * confidence;
            if prediction > 0.25:
                update = prediction - 0.25
                if a in self.failed_actions: confidence = (1 - update) * confidence;
                if a in self.succes_actions: confidence = (1 - update) * confidence + update;
        if self.model.print: print(confidence)
        return confidence

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
            self.c1 = self.update_confidence(self.c1)

        if self.tom_order > 1:
            self.update_b2(f,n)
            self.c2 = self.update_confidence(self.c2)
