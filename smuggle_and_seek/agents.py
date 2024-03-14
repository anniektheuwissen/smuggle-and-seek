import mesa
import random
import numpy as np
from more_itertools import powerset

class Customs(mesa.Agent):
    def __init__(self, unique_id, model, tom_order):
        super().__init__(unique_id, model)
        self.points = 0
        self.action = {}
        self.tom_order = tom_order

        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(set, powerset(np.arange(num_cont))))
        self.b0 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont)

    def step_tom0(self):
        p = 3/2
        for ai in range(len(self.possible_actions)):
            for aj in range(len(self.b0)):
                self.phi[ai] += self.b0[aj] * (10*(aj in self.possible_actions[ai]) - p*len(self.possible_actions[ai]))
            self.phi[ai] = round(self.phi[ai], 4)
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi[1:]),4))[0])]
        print(self.action)

    def step_tom1(self):
        pass

    def step_tom2(self):
        pass
    
    def step(self):
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        elif self.tom_order == 2: self.step_tom2()
        else: print("error")
        # perform action:
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for i in self.action:
                if i == container.unique_id:
                    if (container.num_packages != 0):
                        print(f"CAUGHT {container.num_packages}!!")
                        container.num_packages = 0
                    else:
                        print("wooops caught nothing")
        print(f"hides in container {self.action}")

class Smuggler(mesa.Agent):
    def __init__(self, unique_id, model, tom_order):
        super().__init__(unique_id, model)
        self.points = 0
        self.action = []
        self.tom_order = tom_order
        self.preferences = {}
        self.add_preferences()
        self.num_packages = 0

        #KLOPT NOG HEEL ERG NIET!!
        num_cont = len(self.model.get_agents_of_type(Container))
        self.possible_actions = list(map(set, powerset(np.arange(num_cont))))
        self.b0 = np.array([1/num_cont] * num_cont)
        self.phi = np.zeros(2**num_cont)

    
    def add_preferences(self):
        self.preferences["country"] = self.random.randint(0,2)
        self.preferences["cargo"] = self.random.randint(0,2)

    def new_packages(self):
        self.num_packages += 10

    def step_tom0(self):
        #KLOPT NOG HEEL ERG NIET!!
        n = 3/2
        m = 1/4
        for ai in range(len(self.possible_actions)):
            for aj in range(len(self.b0)):
                none_preferences = 0
                containers = self.model.get_agents_of_type(Container)
                action_ai = self.possible_actions[ai]
                for i in action_ai:
                    for container in containers:
                        if container.unique_id == i:
                            if container.features["cargo"] != self.preferences["cargo"]: none_preferences +=1
                            if container.features["country"] != self.preferences["country"]: none_preferences +=1
                self.phi[ai] += self.b0[aj] * (10*(aj in self.possible_actions[ai]) - n*len(self.possible_actions[ai]) - m*none_preferences)
            self.phi[ai] = round(self.phi[ai], 4)
        print(self.phi)
        self.action = self.possible_actions[random.choice(np.where(self.phi == round(max(self.phi[1:]),4))[0])]
        print(self.action)

    def step_tom1(self):
        pass

    def step(self):
        self.new_packages()
        if self.tom_order == 0: self.step_tom0()
        elif self.tom_order == 1: self.step_tom1()
        else: print("error")
        # perform action:
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            for i in self.action:
                if i == container.unique_id:
                    container.num_packages += 10
        print(f"hides in container {self.action}")
        

class Container(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.features = {}
        self.num_packages = 0

    def add_features(self, x, y):
        self.features["country"] = x
        self.features["cargo"] = y

    def country_feature(self):
        return self.features["country"]

    def cargo_feature(self):
        return self.features["cargo"]