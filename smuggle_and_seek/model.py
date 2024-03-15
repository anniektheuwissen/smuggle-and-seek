import mesa

from .agents import Customs, Smuggler, Container

"""
SmuggleAndSeekGame class: the game in which two agents: a smuggler and a customs can smuggle and seek drugs. The game
environment contains 9 containers, each having 2 features, that the agents can use to hide drugs and seek for drugs.
"""
class SmuggleAndSeekGame(mesa.Model):
    def __init__(self, width, height):
        """
        Initializes the Game
        :param width: The width of the interface
        :param height: The height of the interface
        """
        super().__init__()
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.schedule = mesa.time.BaseScheduler(self)
        self.running = True

        # Add containers to the game, and add features to these containers
        self.num_features = 2
        self.num_cont_per_feat = 2
        x=0; y=0
        for i in range(0,self.num_cont_per_feat**self.num_features):
            container = Container(i, self)
            self.grid.place_agent(container, (x, y))
            container.add_features(x,y)
            if x==self.num_cont_per_feat-1: y+=1; x=0 
            else: x+=1

        # Add agents to the game: one smuggler and one customs
        smuggler = Smuggler(10, self, 0)
        self.schedule.add(smuggler)
        customs = Customs(11, self, 0)
        self.schedule.add(customs)

        # Add datacollector to the game
        self.datacollector = mesa.DataCollector(
            model_reporters= {}, agent_reporters= {}
        )

    def empty_containers(self):
        """
        Removes all drugs from the containers
        """
        for container in self.get_agents_of_type(Container):
            container.num_packages = 0  

    def distribute_points(self, p, n, m):
        """
        Distributes points to the smuggler and customs based on the taken actions.
        """
        # Distribute points to the smuggler based on the amount of successfully smuggled drugs, the amount of 
        # containers used and the amount of features of used containers that were not preferred ones.
        smuggler = self.get_agents_of_type(Smuggler)[0]
        smuggled_drugs = 0; containers_used = len(smuggler.action); none_preferences_used = 0
        for container_id in smuggler.action:
            for container in self.get_agents_of_type(Container):
                if container.unique_id == container_id:
                    smuggled_drugs += container.num_packages
                    none_preferences_used += (container.features["cargo"]!=smuggler.preferences["cargo"])
                    none_preferences_used += (container.features["country"]!=smuggler.preferences["country"])
        smuggler.points += smuggled_drugs - containers_used*n - none_preferences_used*m
        print(f"smuggler's points:{smuggler.points}")
        
        # Distribute points to the customs based on the amount of succesfully caught drugs and the amount of
        # containers checked.
        customs = self.get_agents_of_type(Customs)[0]
        caught_drugs = (10-smuggled_drugs); containers_checked = len(customs.action)
        customs.points += caught_drugs - containers_checked*p
        print(f"customs points:{customs.points}")

    def step(self):
        """
        Performs one step/round/day in which the agents take actions in turn: first the smuggler and then the customs,
        after which points are distributed and the containers are emptied.
        """
        self.datacollector.collect(self)
        self.schedule.step()
        self.distribute_points(0.5, 0.5, 0.25)
        self.empty_containers()