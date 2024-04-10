import mesa

from .agents.customs import Customs
from .agents.smuggler import Smuggler
from .agents.container import Container

def get_average_points_customs(model):
    """
    Returns the average points that customs obtained per day
    """
    if model.day > 0:
        return model.get_agents_of_type(Customs)[0].points / model.day,
    return 0

def get_average_points_smuggler(model):
    """
    Returns the average points that the smuggler obtained per day
    """
    if model.day > 0:
        return model.get_agents_of_type(Smuggler)[0].points / model.day,
    return 0

"""
SmuggleAndSeekGame class: the game in which two agents: a smuggler and a customs can smuggle and seek drugs. The game
environment contains different kinds of9 containers, each having 2 features, that the agents can use to hide drugs and 
seek for drugs.
"""
class SmuggleAndSeekGame(mesa.Model):
    def __init__(self, width, height, tom_customs, tom_smuggler, learning_speed):
        """
        Initializes the Game
        :param width: The width of the interface
        :param height: The height of the interface
        :param tom_customs: The order of theory of mind at which the customs reason
        :param tom_smuggler: The order of theory of mind at which the smuggler reasons
        :param learning_speed: The learning speed of both the customs and smuggler
        """
        super().__init__()
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.schedule = mesa.time.BaseScheduler(self)
        self.running = True

        # Initialize day and packages that are smuggled per day
        self.day = 0
        self.packages_per_day = 5

        # Add containers to the game, add features to these containers, and add container to the grid
        self.num_features = 2; self.num_c_per_feat = 2
        x = 0; y = 0
        for i in range(self.num_c_per_feat**self.num_features):
            container = Container(i, self)
            self.grid.place_agent(container, (x, y))
            self.schedule.add(container)
            container.add_features(x,y)
            if x==self.num_c_per_feat-1: y+=1; x=0 
            else: x+=1

        # Add agents to the game: one smuggler and one customs, and add both to the schedule
        smuggler = Smuggler(i+1, self, tom_smuggler, learning_speed)
        self.schedule.add(smuggler)
        customs = Customs(i+2, self, tom_customs, learning_speed)
        self.schedule.add(customs)

        # Add data collector that collects the points and average points of both the customs and smuggler
        self.datacollector = mesa.DataCollector(
            model_reporters= {
                "customs points": lambda m: m.get_agents_of_type(Customs)[0].points,
                "customs average points": get_average_points_customs,
                "smuggler points": lambda m: m.get_agents_of_type(Smuggler)[0].points,
                "smuggler average points": get_average_points_smuggler,
                }, 
            agent_reporters={
                "used by smugglers": lambda a: getattr(a, "used_s", 0),
                "used by customs": lambda a: getattr(a, "used_c", 0),
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
        Distributes points to the smuggler and customs based on the taken actions.
        """
        # Retrieve the smuggler and customs and their costs parameters
        smuggler = self.get_agents_of_type(Smuggler)[0]
        customs = self.get_agents_of_type(Customs)[0]
        c_s = smuggler.container_costs; f_s = smuggler.feature_costs; c_c = customs.container_costs

        # Distribute points to the smuggler based on the amount of successfully smuggled drugs, the amount of 
        # containers used and the amount of features of used containers that were not preferred ones.
        smuggled_drugs = 0; containers_used = len(smuggler.action); none_preferences_used = 0
        for used_containers in smuggler.action:
            for container in self.get_agents_of_type(Container):
                if container.unique_id == used_containers:
                    smuggled_drugs += container.num_packages
                    none_preferences_used += (container.features["cargo"]!=smuggler.preferences["cargo"]) + (container.features["country"]!=smuggler.preferences["country"])
        smuggler.points += smuggled_drugs - (self.packages_per_day - smuggled_drugs) - c_s*containers_used - f_s*none_preferences_used
        print(f"smuggler's points:{smuggler.points}")
        
        # Distribute points to the customs based on the amount of succesfully caught drugs and the amount of
        # containers checked.
        caught_drugs = (self.packages_per_day - smuggled_drugs); containers_checked = len(customs.action)
        customs.points += caught_drugs - c_c*containers_checked
        print(f"customs points:{customs.points}")

    def agents_update_beliefs(self):
        """
        Lets all agents of type Smuggler and Customs update their beliefs
        """
        self.get_agents_of_type(Customs)[0].update_beliefs()
        self.get_agents_of_type(Smuggler)[0].update_beliefs()


    def step(self):
        """
        Performs one step/round/day in which the agents take actions in turn: first the smuggler and then the customs,
        after which both agents update their beliefs, the points are distributed and the containers are emptied.
        """        
        self.schedule.step()
        self.agents_update_beliefs()
        self.distribute_points()
        self.empty_containers()
        self.day += 1
        self.datacollector.collect(self)
        print("")
        