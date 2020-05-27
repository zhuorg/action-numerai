import numerox as nx
import numerapi
import os
import model

tournaments = nx.tournament_names()
print(tournaments)

data = nx.download('numerai_dataset.zip')
model = nx.logistic()
prediction = nx.backtest(model, data, tournament='bernie', verbosity=1)
logistic(inverse_l2=0.0001)
