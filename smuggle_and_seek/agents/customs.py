import itertools

from .container import Container
from .agent import SmuggleAndSeekAgent

"""
Customs class: the customs agent that tries to capture as many drugs as possible from the containers. They
can have different levels of ToM reasoning.
"""
class Customs(SmuggleAndSeekAgent):
    def __init__(self, unique_id, model, tom_order, learning_speed1, learning_speed2):
        """
        Initializes the agent customs
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed1: The speed at which the agent learns in most situations
        :param learning_speed2: The speed at which the agent learns in less informative situations
        """
        super().__init__(unique_id, model, tom_order, learning_speed1, learning_speed2, "customs")

        self.num_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        self.catched_packages = 0

        self.expected_amount_catch = 1
        self.expected_preferences = self.initialize_expected_preferences()

        num_cont = len(self.model.get_agents_of_type(Container))
        # Define possible actions, and reward and costs vectors
        self.possible_actions = list(map(list, itertools.product([0, 1], repeat=num_cont)))
        self.possible_actions.remove([0]*num_cont)
        self.reward_value = 2
        self.costs_vector = [6] * num_cont

        self.simulationpayoff_o = self.create_simulationpayoff_vector()
        self.simulationpayoff_a = [[self.expected_amount_catch]] * num_cont


    def initialize_expected_preferences(self):
        """
        Initializes expected preferences randomly
        """
        expected_preferences = {}
        for i in range(self.model.num_features):
            expected_preferences[i] = self.random.randint(0,self.model.i_per_feat-1)
        return expected_preferences
    
    def create_simulationpayoff_vector(self):
        """
        Creates the simulation payoff vector
        """
        containers = self.model.get_agents_of_type(Container)
        simulationpayoff = [[self.expected_amount_catch, 0] for _ in range(len(containers))] 
        for idx in range(len(simulationpayoff)):
            simulationpayoff[idx][1] = sum([(containers[idx].features[j] != self.expected_preferences[j]) for j in range(len(self.expected_preferences))]) / len(self.expected_preferences)
        return simulationpayoff

    def take_action(self):
        """
        Performs action and find out succes/failure of action
        """
        if self.model.print: print(f"takes action {self.action}")
        self.failed_actions = []; self.succes_actions = []
        containers = self.model.get_agents_of_type(Container)
        for (c,ai) in enumerate(self.action):
            if ai>0:
                self.num_checks += 1
                containers[c].used_by_c += 1
                if (containers[c].num_packages != 0):
                    if self.model.print: print(f"caught {containers[c].num_packages} packages!!")
                    self.catched_packages += containers[c].num_packages
                    self.succes_actions.append(c)
                    containers[c].num_packages = 0
                    containers[c].used_succ_by_c += 1
                else:
                    if self.model.print: print("wooops caught nothing")
                    self.failed_actions.append(c)
        if self.model.print: print(f"customs succesfull actions are: {self.succes_actions}, and failed actions are: {self.failed_actions}")
        self.successful_checks += len(self.succes_actions)
        self.failed_checks += len(self.failed_actions)
        
    def update_expected_amount_catch(self):
        """
        Updates the expected amount of packages of one catch
        """
        if self.successful_checks > 0:
            self.expected_amount_catch = self.catched_packages / self.successful_checks
            if self.model.print: print(f"expected amount catch is: {self.expected_amount_catch}")

        for i in range(len(self.simulationpayoff_a)): self.simulationpayoff_a[i] = [self.expected_amount_catch]
        self.simulationpayoff_o = self.create_simulationpayoff_vector()

    def update_b0(self):
        """
        Updates b0
        """
        if self.model.print: print("customs are updating beliefs b0 from ... to ...:")
        if self.model.print: print(self.b0)
        if len(self.succes_actions) > 0:
            for c in range(len(self.b0)):
                if c in self.succes_actions:
                    self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + self.learning_speed1
                else: 
                    if c not in self.failed_actions:
                        similarity = 0
                        for cstar in self.succes_actions:
                            similarity += self.similarity(c,cstar)
                        similarity /= len(self.succes_actions)
                        self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + similarity * self.learning_speed1
                    else:
                        self.b0[c] = (1 - self.learning_speed1) * self.b0[c]
        elif len(self.failed_actions) > 0:
            for c in range(len(self.b0)):
                self.b0[c] = (1 - self.learning_speed2) * self.b0[c] + (c not in self.failed_actions) * self.learning_speed2
        if self.model.print: print(self.b0)

    def update_expected_preferences(self):
        """
        Updates the expected preferences of the smuggler
        """
        if self.model.print: print("customs are updating expected preferences beliefs bp to ...:")
        containers = self.model.get_agents_of_type(Container)
        checked = [[0 for _ in range(self.model.i_per_feat)] for _ in range(self.model.num_features)]
        for container in containers:
            for feat in range(len(container.features)):
                checked[feat][container.features[feat]] += container.used_succ_by_c
        for i in range(len(self.expected_preferences)):
            self.expected_preferences[i] = checked[i].index(max(checked[i]))
        if self.model.print: print(f"expected preferences are: {self.expected_preferences}")

    def update_b1(self):
        """
        Updates b1
        """
        if self.model.print: print("customs are updating beliefs b1 from ... to ...:")
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
    
    def update_b2(self):
        """
        Updates b2
        """
        if self.model.print: print("customs are updating beliefs b2 from ... to ...:")
        if self.model.print: print(self.b2)
        if len(self.succes_actions) > 0:
            for c in range(len(self.b2)):
                if c in self.succes_actions:
                    self.b2[c] = (1 - self.learning_speed1) * self.b2[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.succes_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.succes_actions)
                    self.b2[c] = (1 - self.learning_speed1) * self.b2[c] + similarity * self.learning_speed1
        elif len(self.failed_actions) > 0:
            for c in range(len(self.b2)):
                self.b2[c] = (1 - self.learning_speed2) * self.b1[c] + (c not in self.failed_actions) * self.learning_speed2
        if self.model.print: print(self.b2)

    def update_confidence(self, confidence, order):
        """
        Updates confidence (c1 or c2)
        """
        if self.model.print: print("customs are updating confidence from ... to ...:")
        if self.model.print: print(confidence)
        for (c,a) in enumerate(self.action):
            if a>0:
                if order == "1": prediction = self.strategy.prediction_a1[c] / sum(self.strategy.prediction_a1)
                elif order == "2": prediction = self.strategy.prediction_a2[c] / sum(self.strategy.prediction_a2)
                if prediction < 0.25:
                    update = 0.25 - prediction
                    if c in self.failed_actions: confidence = (1 - update) * confidence + update;
                    if c in self.succes_actions: confidence = (1 - update) * confidence;
                if prediction > 0.25:
                    update = prediction - 0.25
                    if c in self.failed_actions: confidence = (1 - update) * confidence;
                    if c in self.succes_actions: confidence = (1 - update) * confidence + update;
        if self.model.print: print(confidence)
        return confidence

    def update_beliefs(self):
        """
        Updates its beliefs, confidence and expectations
        """
        self.update_expected_amount_catch()
        self.update_b0()

        if self.strategy.strategy == "tom1" or self.strategy.strategy == "tom2":
            self.update_expected_preferences()
            self.update_b1()
            self.conf1 = self.update_confidence(self.conf1, "1")

        if self.strategy.strategy == "tom2":
            self.update_b2()
            self.conf2 = self.update_confidence(self.conf2, "2")
    
    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        and taking this action
        """
        # Choose action based on order of tom reasoning
        self.action = self.strategy.choose_action(self.possible_actions, self.b0, self.b1, self.b2, self.conf1, self.conf2, 
                                                  self.reward_value, self.costs_vector, self.expected_amount_catch, self.simulationpayoff_o, self.simulationpayoff_a)

        # Take action
        self.take_action()