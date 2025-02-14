data = [ (7, 4, 4), (9, 6, 6), (6, 6, 4), (4, 6, 7), (7, 5, 5), (9, 5, 7) ]
set_counts = {}
for triplet in data:
    set_counts[triplet] = set_counts.get(triplet, 0) + 1
for set, count in set_counts.items():
    print(f"Set: {set}, Count: {count}")
