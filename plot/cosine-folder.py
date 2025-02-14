import os
def calculate_similarity_index(file1, file2):
  with open(file1, 'r') as f1, open(file2, 'r') as f2:
    text1 = f1.read()
    text2 = f2.read()
  words1 = set(text1.lower().split())
  words2 = set(text2.lower().split())
  common_words = words1.intersection(words2)
  similarity_index = len(common_words) / len(words1)
  return similarity_index
def compare_files_similarity(folder_path):
  file_list = os.listdir(folder_path)
  similarity_matrix = {}
  for file1 in file_list:
    for file2 in file_list:
      if file1 != file2:
        similarity_index = calculate_similarity_index(os.path.join(folder_path, file1), os.path.join(folder_path, file2))
        similarity_matrix[(file1, file2)] = similarity_index
    return similarity_matrix
folder_path = "20"
similarity_matrix = compare_files_similarity(folder_path)
with open('aa.txt', 'w') as np:
  for files, similarity in similarity_matrix.items():
    np.write(f"The similarity index between files {files[0]} and {files[1]} is: {similarity} ")
