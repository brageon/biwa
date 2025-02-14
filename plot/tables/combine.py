from itertools import combinations
data = "6 9 9 9 4 4 6 7 5 6 5 5 4 7 4 9 9 5"
numbers = list(map(int, data.split()))
subsets = set()
for combination in combinations(numbers, 3):
    subsets.add(tuple(combination))
for subset in subsets:
    print(subset)
