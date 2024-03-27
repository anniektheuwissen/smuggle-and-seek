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
    def __init__(self, unique_id, model, tom_order, learning_speed):
        """
        Initializes the agent Customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        """
        super().__init__(unique_id, model, tom_order, learning_speed)

    def step_tom0(self):
        """
        Chooses an action associated with zero-order theory of mind reasoning
        """
        c_c = 1/2

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
        pass

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
        # Choose action
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        elif self.tom_order == 2: self.step_tom2()
        else: print("ERROR: Customs cannot have a theory of mind reasoning above the second order")
        
        # Take action
        print(f"checks containers {self.action}")
        self.failed_actions = []; self.succes_actions = []
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for ai in self.action:
                if ai == container.unique_id:
                    container.checks += 1
                    if (container.num_packages != 0):
                        print(f"caught {container.num_packages} packages!!")
                        container.num_packages = 0
                        self.succes_actions.append(ai)
                    else:
                        print("wooops caught nothing")
                        self.failed_actions.append(ai)
        print(f"customs succesfull actions are: {self.succes_actions}, and failed actions are: {self.failed_actions}")

        #PRINT:
        print("current environment:")
        print([container.num_packages for container in containers])
        
    def update_beliefs(self):
        """
        Updates its beliefs
        """
        # b0
        if self.action == []:
            pass
        else:
            print("customs are updating beliefs from ... to ...:")
            print(self.b0)
            for aj in range(len(self.b0)):
                other_actions_failed_addition = (len(self.failed_actions)/(len(self.b0)-len(self.failed_actions))) * (self.learning_speed/len(self.action))
                succesfull_action_addition = self.learning_speed/len(self.action)
                if aj in self.succes_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + succesfull_action_addition + other_actions_failed_addition
                elif aj in self.failed_actions: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] 
                else: self.b0[aj] = (1 - self.learning_speed) * self.b0[aj] + other_actions_failed_addition
            print(self.b0)