import mesa

from .agents import Customs, Smuggler, Container

class SmuggleAndSeekGame(mesa.Model):
    def __init__(self, width, height):
        super().__init__()
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.schedule = mesa.time.BaseScheduler(self)
        self.running = True

        #Add agents
        smuggler = Smuggler(1, self)
        self.schedule.add(smuggler)
        customs = Customs(0, self)
        self.schedule.add(customs)
        #Add containers
        x=0; y=0
        for i in range(2,11):
            con = Container(i, self)
            # Add the agent to a random grid cell
            self.grid.place_agent(con, (x, y))
            con.add_features(x,y)
            if x==2: y+=1; x=0 
            else: x+=1

        self.datacollector = mesa.DataCollector(
            model_reporters= {}, agent_reporters= {}
        )


    def empty_containers(self):
        for container in self.get_agents_of_type(Container):
            container.num_packages = 0  

    def distribute_points(self, p, n, m):
        #smuggler
        smuggler = self.get_agents_of_type(Smuggler)[0]
        smuggled_drugs = 0; none_preferences_used = 0
        for container_id in smuggler.action:
            for container in self.get_agents_of_type(Container):
                if container.unique_id == container_id:
                    smuggled_drugs += container.num_packages
                    none_preferences_used += (container.features["cargo"]!=smuggler.preferences["cargo"])
                    none_preferences_used += (container.features["country"]!=smuggler.preferences["country"])
        smuggler.points += smuggled_drugs - len(smuggler.action)*n - none_preferences_used*m
        print(f"smuggler's points:{smuggler.points}")
        #customs
        customs = self.get_agents_of_type(Customs)[0]
        customs.points += (10-smuggled_drugs) - len(customs.action)*p
        print(f"customs points:{customs.points}")

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.distribute_points(0.5, 0.5, 0.25)
        self.empty_containers()