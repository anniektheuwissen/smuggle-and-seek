from smuggle_and_seek.model import SmuggleAndSeekGame

model = SmuggleAndSeekGame(10,10)
for i in range(10):
    model.step()


points = model.datacollector.get_agent_vars_dataframe()
print(points)