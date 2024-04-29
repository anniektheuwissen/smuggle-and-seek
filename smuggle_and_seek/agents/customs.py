import random
import numpy as np
from more_itertools import powerset

from .container import Container
from .agent import Agent

"""
Customs class: the customs agent that tries to capture as many drugs as possible from the containers. They
can have different levels of ToM reasoning.
"""
class Customs(Agent):
    def __init__(self, unique_id, model, tom_order, learning_speed, exploration_exploitation):
        """
        Initializes the agent Customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed: The speed at which the agent learns
        """
        super().__init__(unique_id, model, tom_order, learning_speed, exploration_exploitation)
        self.container_costs = 2
        self.num_checks = 0
        self.successful_checks = 0
        self.catched_packages = 0

        self.expected_gain_catch = 1


    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        c_c = self.container_costs

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        for ai in range(len(self.possible_actions)):
            for c in range(len(self.b0)):
                self.phi[ai] += self.b0[c] * (self.expected_gain_catch*(c in self.possible_actions[ai]) - c_c*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        print(f"custom's phi is : {self.phi}")
        softmax_phi = np.exp(self.phi) / np.sum(np.exp(self.phi))
        print(f"customs softmax of phi is : {softmax_phi}")
        action_indexes = [i for i in range(0,len(self.possible_actions))]
        index_action = np.random.choice(action_indexes, 1, p=softmax_phi)[0]
        self.action = self.possible_actions[index_action]

    def step_tom1(self):
        """
        Chooses an action associated with first-order theory of mind reasoning
        """
        print("I am a first order ToM customs")

        c_c = self.container_costs

        # Make prediction about behavior of opponent
        simulation_phi = np.zeros(len(self.b1))
        for c in range(len(self.b1)):
            for c_star in range(len(self.b1)):
                simulation_phi[c] += self.b1[c_star] * (-1*(c == c_star) +1*(c != c_star))
        print(f"custom's simulation phi is : {simulation_phi}")
        # smallest = simulation_phi[0]
        # for i in simulation_phi:
        #     if i < smallest:
        #         smallest = i
        # if smallest < 0:
        #     for i in range(len(simulation_phi)):
        #         simulation_phi[i] += smallest
        #     print(f"updated to.... {simulation_phi}")
        self.prediction_a1 = np.exp(simulation_phi) / np.sum(np.exp(simulation_phi))     
        # for i in range(len(self.prediction_a1)):
        #     if sum(simulation_phi) == 0: self.prediction_a1[i] = 0
        #     else: self.prediction_a1[i] = round(simulation_phi[i] /sum(simulation_phi),2)
        print(f"prediction a1 is : {self.prediction_a1}")

        # Merge prediction with zero-order belief
        W = np.zeros(len(self.b1))
        for c in range(len(self.b1)):
            W[c] = self.c1 * self.prediction_a1[c] + (1-self.c1) * self.b0[c]
        print(f"W is : {W}")

        # Make decision
        # Calculate the subjective value phi for each action, and choose the action with the highest.
        for ai in range(len(self.possible_actions)):
            for c in range(len(W)):
                self.phi[ai] += W[c] * (self.expected_gain_catch*(c in self.possible_actions[ai]) - c_c*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        print(f"custom's phi is : {self.phi}")
        softmax_phi = np.exp(self.phi) / np.sum(np.exp(self.phi))
        print(f"customs softmax of phi is : {softmax_phi}")
        action_indexes = [i for i in range(0,len(self.possible_actions))]
        index_action = np.random.choice(action_indexes, 1, p=softmax_phi)[0]
        self.action = self.possible_actions[index_action]

        # print(f"highest index is at : {np.where(self.phi == max(self.phi))[0]}")
        # self.action = self.possible_actions[random.choice(np.where(self.phi == max(self.phi))[0])]


    def step_tom2(self):
        """
        Chooses an action associated with second-order theory of mind reasoning
        """
        print("I am a second order ToM customs")
        pass
    
    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        and taking this action
        """
        # Reset phi
        for i in range(len(self.phi)): self.phi[i] = 0
        # Choose action based on order of tom reasoning
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        elif self.tom_order == 2: self.step_tom2()
        else: print("ERROR: Customs cannot have a theory of mind reasoning above the second order")
        
        # Random checkups:
        if self.exploration_exploitation:
            if (random.randint(0,99) < 1):
                self.action = random.choice(self.possible_actions)

        # Take action
        print(f"checks containers {self.action}")
        self.failed_actions = []; self.succes_actions = []
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for ai in self.action:
                if ai == container.unique_id:
                    container.used_c += 1
                    if (container.num_packages != 0):
                        print(f"caught {container.num_packages} packages!!")
                        self.catched_packages += container.num_packages
                        container.num_packages = 0
                        self.succes_actions.append(ai)
                    else:
                        print("wooops caught nothing")
                        self.failed_actions.append(ai)
        print(f"customs succesfull actions are: {self.succes_actions}, and failed actions are: {self.failed_actions}")
        self.num_checks += len(self.action); self.successful_checks += len(self.succes_actions)

        #PRINT:
        print("current environment:")
        print([container.num_packages for container in containers])
        
    def common_features(self, c, cstar):
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            if container.unique_id == c:
                feature1_c = container.features["cargo"]
                feature2_c = container.features["country"]
            if container.unique_id == cstar:
                feature1_cstar = container.features["cargo"]
                feature2_cstar = container.features["country"]
        return 0 + (feature1_c == feature1_cstar) + (feature2_c == feature2_cstar)
    
    def uncommon_features(self, c, cstar):
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            if container.unique_id == c:
                feature1_c = container.features["cargo"]
                feature2_c = container.features["country"]
            if container.unique_id == cstar:
                feature1_cstar = container.features["cargo"]
                feature2_cstar = container.features["country"]
        return 0 + (feature1_c != feature1_cstar) + (feature2_c != feature2_cstar)

    def update_beliefs(self):
        """
        Updates its beliefs
        """
        if self.successful_checks > 0:
            self.expected_gain_catch = self.catched_packages / self.successful_checks
            print(f"expected gain catch is: {self.expected_gain_catch}")

        if self.action == []:
            pass
        else:
            f = self.model.num_c_per_feat ** self.model.num_features
            # Update b0
            print("customs are updating beliefs b0 from ... to ...:")
            print(self.b0)
            if len(self.succes_actions) > 0:
                a = (self.learning_speed/f)/len(self.succes_actions)
                for c in range(len(self.b0)):
                    cf_succ = 0
                    for c_star in self.succes_actions:
                        cf_succ += self.common_features(c, c_star)
                    self.b0[c] = (1 - self.learning_speed) * self.b0[c] + a * cf_succ
            elif len(self.failed_actions) > 0:
                a = (self.learning_speed/2/f)/len(self.failed_actions)
                for c in range(len(self.b0)):
                    ucf_fail = 0
                    for c_star in self.failed_actions:
                        ucf_fail += self.common_features(c, c_star)
                    self.b0[c] = (1 - self.learning_speed/2) * self.b0[c] + a * ucf_fail
            # for c in range(len(self.b0)):
            #     cf_succ = 0; ucf_fail = 0; 
            #     for c_star in self.succes_actions:
            #         cf_succ += self.common_features(c, c_star)
            #     for c_star in self.failed_actions:
            #         ucf_fail += self.uncommon_features(c, c_star)
            #     a = (self.learning_speed/f)/len(self.action)
            #     self.b0[c] = (1 - self.learning_speed) * self.b0[c] + a * (cf_succ + ucf_fail)
            print(self.b0)

            if self.tom_order > 0:
                # Update b1
                print("customs are updating beliefs b1 from ... to ...:")
                print(self.b1)
                if len(self.succes_actions) > 0:
                    a = (self.learning_speed/f)/len(self.succes_actions)
                    for c in range(len(self.b1)):
                        cf_succ = 0
                        for c_star in self.succes_actions:
                            cf_succ += self.common_features(c, c_star)
                        self.b1[c] = (1 - self.learning_speed) * self.b1[c] + a * cf_succ
                elif len(self.failed_actions) > 0:
                    b = (self.learning_speed/f)/len(self.failed_actions)
                    for c in range(len(self.b1)):
                        self.b1[c] = (1 - self.learning_speed/f) * self.b1[c] + b * (c in self.failed_actions)
                else: print("stays the same...")
                print(self.b1)

                # Update c1
                update_speed = 0.2

                print("customs are updating c1 from ... to ...:")
                print(self.c1)
                for a in self.action:
                    action_index = self.possible_actions.index(a)
                    update = update_speed * self.prediction_a1[action_index]
                    if a in self.succes_actions: self.c1 = (1 - update) * self.c1 + update * (self.prediction_a1[action_index]/max(self.prediction_a1))
                    if a in self.failed_actions: self.c1 = (1 - update) * self.c1 + update * (1 - self.prediction_a1[action_index]/max(self.prediction_a1))
                print(self.c1)

                # max_indexes_prediction = np.where(self.prediction_a1 == max(self.prediction_a1))[0]
                # print(f"max indexes prediction to update from are: {max_indexes_prediction}")
                # for c in max_indexes_prediction:
                #     if c in self.failed_actions:
                #         self.c1 = (1 - update_speed) * self.c1
                #     elif c in self.succes_actions:
                #         self.c1 = (1 - update_speed) * self.c1 + update_speed
                # if (not any(c in self.action for c in max_indexes_prediction) and len(self.succes_actions) == 0):
                #     self.c1 = (1 - update_speed/3) * self.c1 + update_speed/3
                # print(self.c1)

