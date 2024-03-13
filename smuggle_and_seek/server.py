import mesa

from .agents import Customs, Smuggler, Container
from .model import SmuggleAndSeekGame

def agent_portrayal(agent):
    portrayal = {}

    # if isinstance(agent, Customs):
    #     portrayal["Color"] = "#1ABC9C"
    #     portrayal["Shape"] = "circle"
    #     portrayal["r"] = 0.6
    #     portrayal["Layer"] = 0
    #     portrayal["Filled"] = "true"
    #     portrayal["text"] = "customs"
    #     portrayal["text_color"] = "black"
    # elif isinstance(agent, Smuggler):
    #     portrayal["Color"] = "#E74C3C"
    #     portrayal["Shape"] = "circle"
    #     portrayal["r"] = 0.6
    #     portrayal["Layer"] = 0
    #     portrayal["Filled"] = "true"
    #     portrayal["text"] = "smuggler"
    #     portrayal["text_color"] = "black"
    if isinstance(agent, Container):
        portrayal["Color"] = "#85929E"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Filled"] = "False"
        portrayal["Layer"] = 0
        portrayal["text"] = f"{agent.num_packages}"
        portrayal["text_color"] = "black"
    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 3, 3, 500, 500)
server = mesa.visualization.ModularServer(SmuggleAndSeekGame, 
                           [grid], 
                           "Smuggle and Seek Game", 
                           {"width":3, "height":3})
server.port = 8521 
server.launch()