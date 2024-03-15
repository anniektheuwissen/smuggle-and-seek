import mesa

from .agents import Customs, Smuggler, Container
from .model import SmuggleAndSeekGame

def agent_portrayal(agent):
    """
    Initializes the portrayal of the agents in the visualization
    :param agent: The agent to visualize
    """
    portrayal = {}

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

"""
Add the grid and server, and launch the server
"""
grid = mesa.visualization.CanvasGrid(agent_portrayal, 2, 2, 500, 500)
server = mesa.visualization.ModularServer(SmuggleAndSeekGame, 
                           [grid], 
                           "Smuggle and Seek Game", 
                           {"width":2, "height":2})
server.port = 8521 
server.launch()