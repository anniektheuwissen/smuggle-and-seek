from .container import Container
from .agent import SmuggleAndSeekAgent

"""
Smuggler class: the smuggler agent that tries to smuggle as many drugs as possible through the containers. They
have preferences for certain containers, and they can have different levels of ToM reasoning.
"""
class Smuggler(SmuggleAndSeekAgent):
    def __init__(self, unique_id, model, tom_order, learning_speed1, learning_speed2, packages):
        """
        Initializes the agent Smuggler
        :param unique_id: The unqiue id related to the agent
        :param model: The model in which the agent is placed
        :param tom_order: The order of ToM at which the agent reasons
        :param learning_speed1: The speed at which the agent learns in most situations
        :param learning_speed2: The speed at which the agent learns in less informative situations
        :param packages: The number of packages that the agent has
        """
        super().__init__(unique_id, model, tom_order, learning_speed1, learning_speed2, "smuggler")

        self.preferences = self.add_preferences()
        self.num_packages = packages

        self.num_smuggles = 0
        self.successful_smuggles = 0
        self.successful_smuggled_packages = 0
        self.failed_smuggles = 0
        self.failed_smuggled_packages = 0

        self.average_amount_catch = self.model.packages_per_day
        self.expected_preferences = self.add_preferences()

        num_cont = len(self.model.get_agents_of_type(Container))
        # Define possible actions, and reward and costs vectors
        self.possible_actions = list(self.generate_combinations(packages, num_cont))
        self.reward_value = 2
        self.costs_vector = self.create_costs_vector(6, 1)

        self.simulationpayoff_o = [[self.average_amount_catch]] * num_cont
        self.simulationpayoff_a = self.create_simulationpayoff_vector()
    
    def generate_combinations(self, packages, num_cont):
        """
        Generates all possible combinations of putting the packages in the available containers
        :param packages: The number of packages to distribute
        :param num_cont: The number of containers that are available
        """
        def backtrack(remaining, containers):
            if len(containers) == num_cont:
                if remaining == 0:
                    yield containers
                return
            for i in range(remaining + 1):
                if len(containers) + 1 <= num_cont:
                    yield from backtrack(remaining - i, containers + [i])
        
        return list(backtrack(packages, []))
    
    def create_simulationpayoff_vector(self):
        """
        Creates the simulation payoff vector
        """
        containers = self.model.get_agents_of_type(Container)
        simulationpayoff = [[self.average_amount_catch, 0] for _ in range(len(containers))] 
        for idx in range(len(simulationpayoff)):
            simulationpayoff[idx][1] = sum([(containers[idx].features[j] != self.expected_preferences[j]) for j in range(len(self.expected_preferences))]) / len(self.expected_preferences)
        return simulationpayoff

    def create_costs_vector(self, container_cost, feature_cost):
        """
        Creates the costs vector
        :param container_cost: The costs for using one container
        :param feature_cost: The costs for using a container that has one not preferred feature
        """
        containers = self.model.get_agents_of_type(Container)
        costs_vector = [container_cost] * len(containers)
        for i in range(len(costs_vector)):
            costs_vector[i] += feature_cost * sum([(containers[i].features[j] != self.preferences[j]) for j in range(len(self.preferences))])
        return costs_vector

    def add_preferences(self):
        """
        Assigns random preferences to the smuggler
        """
        preferences = {}
        for i in range(self.model.num_features):
            preferences[i] = self.random.randint(0,self.model.i_per_feat-1)
        if self.model.print: print(f"preferences: {preferences}")
        return preferences


    def take_action(self):
        """
        Performs action
        """
        if self.model.print: print(f"takes action {self.action}")
        for (c, ai) in enumerate(self.action):
            if ai > 0:
                self.model.get_agents_of_type(Container)[c].used_by_s += 1
                self.model.get_agents_of_type(Container)[c].num_packages += ai
                self.num_smuggles += 1

    def check_result_actions(self):
        """
        Checks the results of the taken action, i.e. which actions were successful and which were not
        """
        self.succes_actions = []; self.failed_actions = []
        containers = self.model.get_agents_of_type(Container)
        for (c,ai) in enumerate(self.action):
            if ai>0:
                if containers[c].num_packages == 0: self.failed_actions.append(c); self.failed_smuggled_packages += ai; self.failed_smuggles += 1
                else: self.succes_actions.append(c); self.successful_smuggled_packages += ai ; self.successful_smuggles += 1
        if self.model.print: print(f"smuggler successful actions are: {self.succes_actions}, and failed actions are {self.failed_actions}")
    
    def update_average_amount_per_catch(self):
        """
        Updates the average amount of packages that are catched in one catch
        """
        if (self.num_smuggles - self.successful_smuggles) > 0:
            self.average_amount_catch = self.failed_smuggled_packages / self.failed_smuggles

        for i in range(len(self.simulationpayoff_o)): self.simulationpayoff_o[i] = [self.average_amount_catch]
        self.simulationpayoff_a = self.create_simulationpayoff_vector()

    def update_expected_preferences(self):
        """
        Updates the expected preferences of the smuggler
        """
        if self.model.print: print("smuggler are updating expected preferences beliefs bp from customs to ...:")
        containers = self.model.get_agents_of_type(Container)
        checked = [[0 for _ in range(self.model.i_per_feat)] for _ in range(self.model.num_features)]
        for container in containers:
            for feat in range(len(container.features)):
                checked[feat][container.features[feat]] += container.used_succ_by_c
        for i in range(len(self.expected_preferences)):
            self.expected_preferences[i] = checked[i].index(max(checked[i]))
        if self.model.print: print(f"expected preferences are: {self.expected_preferences}")
        
    def update_b0(self):
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
                    if c not in self.succes_actions:
                        similarity = 0
                        for cstar in self.failed_actions:
                            similarity += self.similarity(c,cstar)
                        similarity /= len(self.failed_actions)
                        self.b0[c] = (1 - self.learning_speed1) * self.b0[c] + similarity * self.learning_speed1
                    else:
                        self.b0[c] = (1 - self.learning_speed1) * self.b0[c]
        elif len(self.succes_actions) > 0:
            for c in range(len(self.b0)):
                self.b0[c] = (1 - self.learning_speed2) * self.b0[c] + (c not in self.succes_actions) * self.learning_speed2
        if self.model.print: print(self.b0)

    def update_b1(self):
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

    def update_b2(self):
        """
        Updates b2
        """
        if self.model.print: print("smuggler is updating beliefs b2 from ... to ...:")
        if self.model.print: print(self.b2)
        if len(self.failed_actions) > 0:
            for c in range(len(self.b2)):
                if c in self.failed_actions:
                    self.b2[c] = (1 - self.learning_speed1) * self.b2[c] + self.learning_speed1
                else: 
                    similarity = 0
                    for cstar in self.failed_actions:
                        similarity += self.similarity(c,cstar)
                    similarity /= len(self.failed_actions)
                    self.b2[c] = (1 - self.learning_speed1) * self.b2[c] + similarity * self.learning_speed1
        elif len(self.succes_actions) > 0:
            for c in range(len(self.b2)):
                self.b2[c] = (1 - self.learning_speed2) * self.b1[c] + (c not in self.succes_actions) * self.learning_speed2
        if self.model.print: print(self.b2)
    
    def update_confidence(self, confidence, order):
        """
        Updates confidence (c1 or c2)
        """
        if self.model.print: print("smuggler is updating confidence from ... to ...:")
        if self.model.print: print(confidence)
        for (c,a) in enumerate(self.action):
            if a>0:
                if order == "1": prediction = self.strategy.prediction_a1[c] / sum(self.strategy.prediction_a1)
                elif order == "2": prediction = self.strategy.prediction_a2[c] / sum(self.strategy.prediction_a2)
                if prediction < 0.25:
                    update = 0.25 - prediction
                    if c in self.succes_actions:
                        confidence = (1 - update) * confidence + update;
                    if c in self.failed_actions: confidence = (1 - update) * confidence;
                if prediction > 0.25:
                    update = prediction - 0.25
                    if c in self.succes_actions: 
                        confidence = (1 - update) * confidence;
                    if c in self.failed_actions: confidence = (1 - update) * confidence + update;
        if self.model.print: print(confidence)
        return confidence

    def update_beliefs(self):
        """
        Updates its beliefs and confidence
        """
        self.check_result_actions()
        self.update_average_amount_per_catch()
        self.update_b0()

        if self.strategy.strategy == "tom1" or self.strategy.strategy == "tom2":
            self.update_b1()
            self.conf1 = self.update_confidence(self.conf1, "1")
        
        if self.strategy.strategy == "tom2":
            self.update_b2()
            self.conf2 = self.update_confidence(self.conf2, "2")
            self.update_expected_preferences()

    def step(self):
        """
        Performs one step by choosing an action associated with its order of theory of mind reasoning,
        and taking this action
        """
        # Choose action based on order of tom reasoning
        self.action = self.strategy.choose_action(self.possible_actions, self.b0, self.b1, self.b2, self.conf1, self.conf2, 
                                                  self.reward_value, self.costs_vector, self.average_amount_catch, self.simulationpayoff_o, self.simulationpayoff_a)

        # Take action
        self.take_action()