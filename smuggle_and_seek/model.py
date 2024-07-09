import mesa
import numpy as np

from .agents.police import Police
from .agents.smuggler import Smuggler
from .agents.container import Container

"""
SmuggleAndSeekGame class: the game in which two agents: a smuggler and a police can smuggle and seek drugs. The game
environment contains different kinds of containers, each having 2 features, that the agents can use to hide drugs and 
seek for drugs.
"""
class SmuggleAndSeekGame(mesa.Model):
    def __init__(self, k, l, m, tom_police, tom_smuggler, learning_speed1, learning_speed2):
        """
        Initializes the Game
        :param width: The width of the interface
        :param height: The height of the interface
        :param tom_police: The order of theory of mind at which the police reason
        :param tom_smuggler: The order of theory of mind at which the smuggler reasons
        :param learning_speed1: The learning speed at which both the police and smuggler learn in most situations
        :param learning_speed2: The learning speed at which both the police and smuggler learn in less informative situations
        """
        super().__init__()
        self.print = False
        
        # Initialize grid and schedules
        self.grid = mesa.space.MultiGrid(l, l, True)
        self.running_schedule = mesa.time.BaseScheduler(self)
        self.schedule = mesa.time.BaseScheduler(self)
        self.running = True

        # Initialize day and packages that are smuggled per day
        self.day = 0
        self.packages_per_day = m

        # Add containers to the game, add features to these containers, and add container to the grid
        self.num_features = k; self.i_per_feat = l
        features = [0]*self.num_features
        for i in range(self.i_per_feat**self.num_features):
            container = Container(i, self)
            container.add_features(features)
            self.grid.place_agent(container, (features[-1], features[-2]))
            self.schedule.add(container)
            if features[-1] == (self.i_per_feat-1):
                for j in range(0,self.num_features):
                    if features[-(1+j)] != (self.i_per_feat-1): features[-(1+j)] += 1; features[-(j):] = [0] * len(features[-(j):]); break
            else: features[-1] += 1  

        # Add agents to the game: one smuggler and one police, and add both to the running schedule
        smuggler = Smuggler(i+1, self, tom_smuggler, learning_speed1, learning_speed2, m)
        self.running_schedule.add(smuggler)
        police = Police(i+2, self, tom_police, learning_speed1, learning_speed2)
        self.running_schedule.add(police)

        # Add data collector that collects the points and average points of both the police and smuggler
        self.datacollector = mesa.DataCollector(
            model_reporters= {
                "police points": lambda m: m.get_agents_of_type(Police)[0].points,
                "smuggler points": lambda m: m.get_agents_of_type(Smuggler)[0].points,
                "police points averaged": lambda m: sum(m.get_agents_of_type(Police)[0].points_queue) / 10,
                "smuggler points averaged": lambda m: sum(m.get_agents_of_type(Smuggler)[0].points_queue) / 10,
                "successful checks": lambda m: m.get_agents_of_type(Police)[0].successful_checks,
                "successful smuggles": lambda m: m.get_agents_of_type(Smuggler)[0].successful_smuggles,
                "caught packages": lambda m: m.get_agents_of_type(Police)[0].catched_packages,
                "smuggled packages": lambda m: m.get_agents_of_type(Smuggler)[0].successful_smuggled_packages,
                }, 
            agent_reporters={
                "used by smugglers": lambda a: getattr(a, "used_by_s", 0),
                "used by police": lambda a: getattr(a, "used_by_c", 0),
                },
        )
        self.datacollector.collect(self)

    def empty_containers(self):
        """
        Removes all drugs from the containers (called at the end of the day)
        """
        for container in self.get_agents_of_type(Container):
            container.num_packages = 0  

    def distribute_points(self):
        """
        Distributes points to the smuggler and police based on the taken actions.
        """
        # Retrieve the smuggler and police and their costs parameters
        smuggler = self.get_agents_of_type(Smuggler)[0]
        police = self.get_agents_of_type(Police)[0]

        smuggler_reward = smuggler.reward_value * (self.packages_per_day - np.dot(smuggler.action, police.action)) - np.dot(smuggler.costs_vector,[int(c>0) for c in smuggler.action])
        smuggler.points += int(smuggler_reward)
        smuggler.points_queue.pop(0); smuggler.points_queue.append(smuggler_reward)
        if self.print: print(f"smuggler's points:{smuggler.points}")
        
        police_reward = (police.reward_value/police.expected_amount_catch) * np.dot(police.action, smuggler.action) - np.dot(police.costs_vector,police.action)
        police.points += int(police_reward)
        police.points_queue.pop(0); police.points_queue.append(police_reward)
        if self.print: print(f"police's points:{police.points}")

    def agents_update_beliefs(self):
        """
        Lets all agents of type Smuggler and police update their beliefs
        """
        for agent in self.get_agents_of_type(Police): agent.update_beliefs()
        for agent in self.get_agents_of_type(Smuggler): agent.update_beliefs()

    def step(self):
        """
        Performs one step/round/day in which the agents take actions in turn: first the smuggler and then the police,
        after which the points are distributed, both agents update their beliefs, and the containers are emptied.
        """ 
        self.get_agents_of_type(Smuggler)[0].num_packages = self.packages_per_day    
        self.running_schedule.step()
        self.distribute_points()
        self.agents_update_beliefs()
        self.empty_containers()
        self.day += 1
        self.datacollector.collect(self)

        if self.print: print("")

        # To be able to run a batch run:
        if self.day == 365: self.running = False
        