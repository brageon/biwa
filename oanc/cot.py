def count_matching_pairs(text):
  pairs = {}
  for i in range(len(text) - 1):
    pair = text[i] + text[i + 1]
    if pair in pairs:
      pairs[pair] += 1
    else:
      pairs[pair] = 1
  return pairs
target_pairs = ["DN", "ND", "CD", "DC", "DD", "NC", "CN", "NN", "CC", "HH", "DH", "HD", "CH", "HC", "HN", "NH"]
filename = 'sort.txt'
with open(filename, 'r') as file:
    text = file.read()

all_pairs = count_matching_pairs(text)
pair_counts = {pair: all_pairs.get(pair, 0) for pair in target_pairs}

for pair, count in pair_counts.items():
  print(f"Pair '{pair}': {count}")
"""
def separate_by_dots(filename):
  with open(filename, 'r') as f:
    text = f.read()
  new_text = text.replace(".", ".\n")
  with open(filename, 'w') as f:
    f.write(new_text)

filename = "tree.txt"
separate_by_dots(filename)

with open(filename, 'r') as f:
  modified_text = f.read()
  print(modified_text)
"""
