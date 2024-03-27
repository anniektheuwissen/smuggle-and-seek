import mesa

from .model import SmuggleAndSeekGame, Smuggler, Customs, Container

def color_variant(hex_color, brightness_offset=1):
    """ takes a color like #87c95f and produces a lighter or darker variant """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])

def portrayal_customs_grid(agent):
    """
    Initializes the portrayal of the agents in the visualization
    :param agent: The agent to visualize
    """
    portrayal = {}

    if isinstance(agent, Container):
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Filled"] = "False"
        portrayal["Layer"] = 0
        portrayal["text"] = f"country:{agent.features["country"]}, cargo:{agent.features["cargo"]}"
        portrayal["text_color"] = "black"

        if agent.model.day > 0:
            portrayal["Color"] = color_variant("#D5F5E3", int(-50 + 100*(agent.checks / agent.model.day)))
        else:
            portrayal["Color"] = "#D5F5E3"

        if agent.model.day > 0:
            print(f"{agent.unique_id}, {agent.features["country"], agent.features["cargo"]} : {agent.checks / agent.model.day}")
        
    return portrayal

def portrayal_smuggler_grid(agent):
    """
    Initializes the portrayal of the agents in the visualization
    :param agent: The agent to visualize
    """
    portrayal = {}

    if isinstance(agent, Container):
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Filled"] = "False"
        portrayal["Layer"] = 0
        portrayal["text"] = f"country:{agent.features["country"]}, cargo:{agent.features["cargo"]}"
        portrayal["text_color"] = "black"

        if agent.model.day > 0:
            portrayal["Color"] = color_variant("#FADBD8", int(-50 + 100*(agent.smuggles / agent.model.day)))
        else:
            portrayal["Color"] = "#FADBD8"

        if agent.model.day > 0:
            print(f"{agent.unique_id}, {agent.features["country"], agent.features["cargo"]} : {agent.smuggles / agent.model.day}")
        
    return portrayal

def customs_grid_name(model):
    """
    Display a text representing the name of the grid.
    """
    last_action = model.get_agents_of_type(Customs)[0].action
    tab = "&nbsp &nbsp &nbsp &nbsp"   
    return f"Customs distribution of actions: {tab}{tab}{tab} last action:{last_action}"

def smuggler_grid_name(model):
    """
    Display a text representing the name of the grid.
    """
    last_action = model.get_agents_of_type(Smuggler)[0].action
    tab = "&nbsp &nbsp &nbsp &nbsp"   
    return f"Smuggler's distribution of actions: {tab}{tab}&nbsp &nbsp last action:{last_action}"

def chart_name1(model):
    """
    Display a text representing the name of the chart.
    """
    return f"Points:"

def chart_name2(model):
    """
    Display a text representing the name of the chart.
    """
    return f"Average points:"

"""
Add the grid and server, and launch the server
"""
grid1 = mesa.visualization.CanvasGrid(portrayal_customs_grid, 2, 2, 500, 500)

grid2 = mesa.visualization.CanvasGrid(portrayal_smuggler_grid, 2, 2, 500, 500)

chart1 = mesa.visualization.ChartModule(
    [
        {"Label": "customs points", "Color": "#a3c3b1"},
        {"Label": "smuggler points", "Color": "#c8a9a6"},
    ]
)

chart2 = mesa.visualization.ChartModule(
    [
        {"Label": "customs average points", "Color": "#a3c3b1"},
        {"Label": "smuggler average points", "Color": "#c8a9a6"},
    ]
)

model_params = {
    "width": 2,
    "height": 2,
    "tom_order_customs": mesa.visualization.Choice(
        "Customs ToM order",
        value=0,
        choices=[0,1,2],
    ),
    "tom_order_smuggler": mesa.visualization.Choice(
        "Smuggler ToM order",
        value=0,
        choices=[0,1],
    ),
    "learning_speed": mesa.visualization.Slider(
        "Learning speed",
        value=0.2,
        min_value=0,
        max_value=1,
        step=0.1
    ),
}

server = mesa.visualization.ModularServer(SmuggleAndSeekGame, 
                           [customs_grid_name, grid1, smuggler_grid_name, grid2, chart_name1, chart1, chart_name2, chart2], 
                           "Smuggle and Seek Game", 
                           model_params)
server.port = 8521