import mesa

class Customs(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.points = 0
        self.action = []

    def randomly_check(self):
        container_id = self.random.randint(2,10)
        print(f"checking container {container_id}")
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            if container_id == container.unique_id:
                if (container.num_packages != 0):
                    print(f"CAUGHT {container.num_packages}!!")
                    container.num_packages = 0
                else:
                    print("wooops caught nothing")
        return [container_id]
    
    def step(self):
        self.action = self.randomly_check()

class Smuggler(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.points = 0
        self.action = []
        self.preferences = {}
        self.add_preferences()
        self.num_packages = 0
    
    def add_preferences(self):
        self.preferences["country"] = self.random.randint(0,2)
        self.preferences["cargo"] = self.random.randint(0,2)

    def new_packages(self):
        self.num_packages += 10

    def randomly_hide_all(self):
        container_id = self.random.randint(2,10)
        containers = self.model.get_agents_of_type(Container)
        for container in containers:
            if container_id == container.unique_id:
                container.num_packages += 10
        print(f"hides in container {container_id}")
        return [container_id]
    
    def step(self):
        self.new_packages()
        self.action = self.randomly_hide_all()
        

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