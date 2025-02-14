import pprint, pickle

def list_pickle_methods(pickle_file):
  try:
    with open(pickle_file, "rb") as f:
      obj = pickle.load(f)
    methods = [method for method in dir(obj) if not method.startswith("__")]
    print(f"Methods available on the loaded object:")
    for method in methods:
      print(method)
  except (AttributeError, FileNotFoundError) as e:
    print(f"Error: {e}")

pickle_file = "averaged_perceptron_tagger.pickle" 
list_pickle_methods(pickle_file)

with open('averaged_perceptron_tagger.pickle', 'rb') as f:
    tokens = pickle.loads(f.read()) 
pprint.pprint(tokens)