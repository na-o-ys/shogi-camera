from ._util import save_model
from ._purple import learn as purple
from ._purple2 import learn as purple2
from ._blue import learn as blue
from ._blue2 import learn as blue2
from ._blue3 import learn as blue3
from ._yellow import learn as yellow

models = {
   "purple": purple,
   "purple2": purple2,
   "blue": blue,
   "blue2": blue2,
   "blue3": blue3,
   "yellow": yellow
}

def learn_model(name, data_dir, verbose=True):
   return models[name](data_dir, verbose)
