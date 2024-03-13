import mesa

from .agents import Customs, Smuggler, Container
from .model import SmuggleAndSeekGame

def agent_portrayal(agent):
    portrayal = {}

    if isinstance(agent, Customs):
        portrayal["Color"] = "grey"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 1
        portrayal["Filled"] = "true"
    elif isinstance(agent, Smuggler):
        portrayal["Color"] = "red"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 1
        portrayal["Filled"] = "true"
    elif isinstance(agent, Container):
        portrayal["Color"] = "blue"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Layer"] = 1
        portrayal["Filled"] = "true"
    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 7, 7, 500, 500)
server = mesa.visualization.ModularServer(SmuggleAndSeekGame, 
                           [grid], 
                           "Smuggle and Seek Game", 
                           {"width":7, "height":7})
server.port = 8521 
server.launch()