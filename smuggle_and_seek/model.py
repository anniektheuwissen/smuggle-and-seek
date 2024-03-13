import mesa

from .agents import Customs, Smuggler, Container

class SmuggleAndSeekGame(mesa.Model):

    def __init__(self, width, height):
        super().__init__()
        self.num_agents = 2
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        #Add agents
        c = Customs(0, self)
        self.schedule.add(c)
        self.grid.place_agent(c, (0,int(height/2)))
        s = Smuggler(1, self)
        self.schedule.add(s)
        self.grid.place_agent(s, (width-1,int(height/2)))
        #Add containers
        x=2; y=2
        for i in range(2,11):
            con = Container(i, self)
            self.schedule.add(con)
            # Add the agent to a random grid cell
            self.grid.place_agent(con, (x, y))
            if x==4: y+=1; x=2 
            else: x+=1


        self.datacollector = mesa.DataCollector(
            model_reporters= {"Num_agents": "num_agents"}, agent_reporters= {"Age": "age"}
        )
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()