from collections import defaultdict
KD = '9 4 7 6 5 4 4 4 5 5'
DE = '6 4 4 6 6 7 5 4 6 6 8 8 5 5 7 4 7 6 6 4 6 4 6 4 4 4 5 4 5 5 2 3 7 5 6 6 6 7 1 2 4 7 6 6 6 5 6 5 8 4 6 4 8 5 7 4 3 5 5 6 5 4 4 6 9 8 5 8 5 6 6 7 5 6 7 4 6 6 5 5 7 8 9 7 6 4 3 9 6 5 5 6 5 4 5 9 4 4 2 6 8'
def jaccard_similarity(s1, s2):
  s1_set = set(s1)
  s2_set = set(s2)
  intersection = s1_set & s2_set
  union = s1_set | s2_set
  return len(intersection) / len(union)
best_matches = defaultdict(list)
best_similarity = 0
for i in range(len(DE)):
  for j in range(i + 1, len(DE) + 1):
    substring = DE[i:j]
    similarity = jaccard_similarity(substring, KD)
    #similarity = levenshtein_distance(substring, KD)
    if similarity > best_similarity:
      best_similarity = similarity
      best_matches.clear()
      best_matches[similarity].append(substring)
      best_matches[similarity].append(i)
    elif similarity == best_similarity:
      best_matches[similarity].append(substring)
      best_matches[similarity].append(i)
for similarity, matches in best_matches.items():
  for substring, position in zip(matches[::2], matches[1::2]):
    print(f"Similarity: {similarity:.3f}")
    print(f"Substring: {substring}")
    print(f"Position: {position}")
    print("=" * 20)
