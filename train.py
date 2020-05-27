import numerox as nx
import numerapi
import os
import model

tournaments = nx.tournament_names()
print(tournaments)

data = nx.download('numerai_dataset.zip', load=False)

print('hello')
