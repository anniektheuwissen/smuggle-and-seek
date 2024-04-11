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
        self.container_costs = 1/2
        self.num_checks = 0
        self.successful_checks = 0


    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        c_c = self.container_costs

        # Calculate the subjective value phi for each action, and choose the action with the highest.
        for ai in range(len(self.possible_actions)):
            for c in range(len(self.b0)):
                self.phi[ai] += self.b0[c] * (1*(c in self.possible_actions[ai]) - c_c*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        print(f"custom's phi is : {self.phi}")
        print(f"highest index is at : {np.where(self.phi == round(max(self.phi),4))[0]}")
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi),4))[0])]

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
        smallest = simulation_phi[0]
        for i in simulation_phi:
            if i < smallest:
                smallest = i
        if smallest < 0:
            for i in range(len(simulation_phi)):
                simulation_phi[i] += smallest
            print(f"updated to.... {simulation_phi}")
        for i in range(len(self.prediction_a1)):
            self.prediction_a1[i] = simulation_phi[i] /sum(simulation_phi)
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
                self.phi[ai] += W[c] * (1*(c in self.possible_actions[ai]) - c_c*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        print(f"custom's phi is : {self.phi}")
        print(f"highest index is at : {np.where(self.phi == round(max(self.phi),4))[0]}")
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi),4))[0])]


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
        
    def update_beliefs(self):
        """
        Updates its beliefs
        """
        if self.action == []:
            pass
        else:
            # Update b0
            print("customs are updating beliefs b0 from ... to ...:")
            print(self.b0)
            for aj in range(len(self.b0)):
                other_actions_failed_addition = (len(self.failed_actions)/(len(self.b0)-len(self.failed_actions))) * (self.learning_speed/len(self.action))
                succesfull_action_addition = self.learning_speed/len(self.action)
                if aj in self.succes_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + succesfull_action_addition + other_actions_failed_addition
                elif aj in self.failed_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] 
                else: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + other_actions_failed_addition
            print(self.b0)

            if self.tom_order > 0:
                # Update b1
                print("customs are updating beliefs b1 from ... to ...:")
                print(self.b1)
                if len(self.succes_actions) > 0:
                    for aj in range(len(self.b1)):
                        succesfull_action_addition = self.learning_speed/len(self.succes_actions)
                        if aj in self.succes_actions: self.b1[aj] = (1 - self.learning_speed) * self.b1[aj] + succesfull_action_addition
                        else: self.b1[aj] = (1 - self.learning_speed) * self.b1[aj]
                else: print("stays the same...")
                print(self.b1)

                # Update c1
                update_speed = 0.2

                print("customs are updating c1 from ... to ...:")
                print(self.c1)
                max_indexes_prediction = np.where(self.prediction_a1 == max(self.prediction_a1))[0]
                print(f"max indexes prediction to update from are: {max_indexes_prediction}")
                for c in max_indexes_prediction:
                    if c in self.failed_actions:
                        self.c1 = (1 - update_speed) * self.c1
                    elif c in self.succes_actions:
                        self.c1 = (1 - update_speed) * self.c1 + update_speed
                print(self.c1)

