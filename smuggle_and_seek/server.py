import mesa

from .model import SmuggleAndSeekGame, Smuggler, Customs, Container

"""
The server
"""

# Adjust this variable when you want to adjust the grid size:
############################################################
l = 2
############################################################

def color_variant(hex_color, brightness_offset=1):
    """ 
    Takes a color like #87c95f and produces a lighter or darker variant 
    (taken from https://chase-seibert.github.io/blog/2011/07/29/python-calculate-lighterdarker-rgb-colors.html)
    """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])

def portrayal_customs_grid(agent):
    """
    Initializes the portrayal of the agents in the visualization of the customs grid
    :param agent: The agent to visualize
    """
    portrayal = {}

    if isinstance(agent, Container):
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Filled"] = "False"
        portrayal["Layer"] = 0
        portrayal["text"] = f"{agent.unique_id}: (country:{agent.features[0]}, cargo:{agent.features[1]})"
        portrayal["text_color"] = "black"
        portrayal["size"] = 100

        total_containers_used = 0
        for container in agent.model.get_agents_of_type(Container): total_containers_used += container.used_by_c

        # Makes the color darker/brighter corresponding to whether the percentage that it is chosen as action 
        # (low percentage = dark, high percentage = bright)
        if agent.model.day > 0:
            portrayal["Color"] = color_variant("#D5F5E3", int(-50 + 100*(agent.used_by_c / total_containers_used)))
        else:
            portrayal["Color"] = "#D5F5E3"

    return portrayal

def portrayal_smuggler_grid(agent):
    """
    Initializes the portrayal of the agents in the visualization of the smuggler grid
    :param agent: The agent to visualize
    """
    portrayal = {}

    if isinstance(agent, Container):
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Filled"] = "False"
        portrayal["Layer"] = 0
        portrayal["text"] = f"{agent.unique_id}: (country:{agent.features[0]}, cargo:{agent.features[1]})"
        portrayal["text_color"] = "black"

        total_containers_used = 0
        for container in agent.model.get_agents_of_type(Container): total_containers_used += container.used_by_s
        
        # Makes the color darker/brighter corresponding to whether the percentage that it is chosen as action 
        # (low percentage = dark, high percentage = bright)
        if agent.model.day > 0:
            portrayal["Color"] = color_variant("#FADBD8", int(-50 + 100*(agent.used_by_s / total_containers_used)))
        else:
            portrayal["Color"] = "#FADBD8"

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

def barchart_name(model):
    """
    Display a text representing the name of the grid.
    """
    return f"Smuggler's and customs distribution of actions:"

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

def preferences(model):
    """
    Display a text representing the preffered features of the smuggler
    """
    preferences = model.get_agents_of_type(Smuggler)[0].preferences
    return f"Preferences: (country: {preferences[0]}, cargo: {preferences[1]})"

def succesfull_checks(model):
    """
    Display a text representing the succesful checks
    """
    successful_checks = model.datacollector.get_model_vars_dataframe()['successful checks'][model.day]
    num_checks = model.get_agents_of_type(Customs)[0].num_checks
    if (num_checks > 0): percentage = round(successful_checks/num_checks * 100,2)
    else: percentage = 0
    tab = "&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp"  
    return f"Number of successful checks: {successful_checks} {tab}{tab}{tab}{tab} percentage: {percentage} %"

def succesfull_smuggles(model):
    """
    Display a text representing the succesful smuggles
    """
    successful_smuggles = model.datacollector.get_model_vars_dataframe()['successful smuggles'][model.day]
    num_smuggles = model.get_agents_of_type(Smuggler)[0].num_smuggles
    if (num_smuggles > 0): percentage = round(successful_smuggles/num_smuggles * 100,2)
    else: percentage = 0
    tab = "&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp"  
    return f"Number of successful smuggles: {successful_smuggles} {tab}{tab}{tab}&nbsp &nbsp &nbsp &nbsp &nbsp&nbsp percentage: {percentage} %"

def caught_packages(model):
    """
    Display a text representing the caught packages
    """
    amount = model.datacollector.get_model_vars_dataframe()['caught packages'][model.day]
    num_packages = model.packages_per_day * model.day
    if (num_packages> 0): percentage = round(amount/num_packages * 100,2)
    else: percentage = 0
    tab = "&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp"  
    return f"Number of caught packages: {amount} {tab}{tab}{tab}{tab} percentage: {percentage} %"

def smuggled_packages(model):
    """
    Display a text representing the smuggled packages
    """
    amount = model.datacollector.get_model_vars_dataframe()['smuggled packages'][model.day]
    num_packages = model.packages_per_day * model.day
    if (num_packages> 0): percentage = round(amount/num_packages * 100,2)
    else: percentage = 0
    tab = "&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp"  
    return f"Number of smuggled packages: {amount} {tab}{tab}{tab}&nbsp &nbsp &nbsp &nbsp &nbsp&nbsp percentage: {percentage} %"


"""
Add the grids, charts, model parameters and server
"""
grid1 = mesa.visualization.CanvasGrid(portrayal_customs_grid, l, l, 500, 500)
grid2 = mesa.visualization.CanvasGrid(portrayal_smuggler_grid, l, l, 500, 500)

barchart = mesa.visualization.BarChartModule(
    [
        {"Label": "used by smugglers", "Color": "#c8a9a6"},
        {"Label": "used by customs", "Color": "#a3c3b1"},
    ],
    scope="agent" 
)

chart1 = mesa.visualization.ChartModule(
    [
        {"Label": "customs points", "Color": "#a3c3b1"},
        {"Label": "smuggler points", "Color": "#c8a9a6"},
    ],
)

chart2 = mesa.visualization.ChartModule(
    [
        {"Label": "customs points averaged", "Color": "#a3c3b1"},
        {"Label": "smuggler points averaged", "Color": "#c8a9a6"},
    ],
)

model_params = {
    "k": 2,
    "l": l,
    "m": mesa.visualization.Slider(
        "m",
        value=5,
        min_value=1,
        max_value=10,
        step=1
    ),
    "tom_customs": mesa.visualization.Choice(
        "customs ToM order",
        value=0,
        choices=[0,1,2],
    ),
    "tom_smuggler": mesa.visualization.Choice(
        "Smuggler ToM order",
        value=0,
        choices=[0,1,2],
    ),
    "learning_speed1": mesa.visualization.Slider(
        "Learning speed1",
        value=0.2,
        min_value=0,
        max_value=1,
        step=0.1
    ),
    "learning_speed2": mesa.visualization.Slider(
        "Learning speed2",
        value=0.05,
        min_value=0,
        max_value=1,
        step=0.01
    ),
}


server = mesa.visualization.ModularServer(SmuggleAndSeekGame, 
                           [smuggler_grid_name, grid2, customs_grid_name, grid1, barchart_name, barchart, preferences, succesfull_smuggles, succesfull_checks, smuggled_packages, caught_packages, chart_name1, chart1, chart_name2, chart2], 
                           "Smuggle and Seek Game", 
                           model_params)
server.port = 8521