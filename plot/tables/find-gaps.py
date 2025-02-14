'''
def find_gaps(data):
    gaps = []
    for i in range(len(data) - 1):
        current_tuple = data[i]
        next_tuple = data[i + 1]
        gap = next_tuple[1] - current_tuple[1]
        gaps.append((current_tuple, next_tuple, gap))
    return gaps
def find_gaps(data):
    max_gap = None
    max_gap_tuples = []
    for i in range(len(data) - 1):
        current_tuple = data[i]
        next_tuple = data[i + 1]
        gap = next_tuple[1] - current_tuple[1]
        if max_gap is None or gap > max_gap:
            max_gap = gap
            max_gap_tuples = []
        if gap == max_gap:
            max_gap_tuples.append((current_tuple, next_tuple))
    return max_gap, max_gap_tuples
'''
def find_gaps(data):
    max_gap = None
    max_gap_tuples = []
    for i in range(len(data) - 1):
        current_tuple = data[i]
        next_tuple = data[i + 1]
        gap = next_tuple[1] - current_tuple[1]
        if max_gap is None or gap > max_gap:
            if gap == 24: # 12 gaps because of whitespace 
                max_gap = gap
                max_gap_tuples = []
            else:
                continue
        if gap == max_gap:
            modicurr = (current_tuple[1] // 2)
            modified = (next_tuple[1] // 2)
            max_gap_tuples.append((modicurr, modified))
    return max_gap, max_gap_tuples 
data = [(1, 1), (1, 19), (1, 21), (1, 23), (1, 35), (1, 49), (1, 51), (1, 111), (1, 115), (1, 129), (1, 131), (1, 8873)]        
sova = find_gaps(data)
print(sova)
