import mesa

class Customs(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = 0
    
    def step(self):
        #...
        pass

class Smuggler(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = 0
        self.preferences = []
    
    def step(self):
        #...
        pass


class Container(mesa.Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.features = {}

    def add_features(self, type, value):
        self.features[type]=value
    
    def step(self):
        #...
        pass