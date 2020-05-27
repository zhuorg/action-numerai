import numerox as nx
import numerapi
import os
import model

tournaments = nx.tournament_names()
print(tournaments)
try
  # download dataset from numerai
  data = nx.download('numerai_dataset.zip')
except OSError as err:
    print("OS error: {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
for tournament_name in tournaments:
    # create your model
    m = model.LinearModel(verbose=True)

    print("fitting model for", tournament_name)
    m.fit(data['train'], tournament_name)

    print("saving model for", tournament_name)
    m.save('model_trained_' + tournament_name)
