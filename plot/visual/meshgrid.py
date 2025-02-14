import numpy as np
import matplotlib.pyplot as plt
def load_matrix(file_path):
  with open(file_path, "r") as f:
    data = f.readlines()
    matrix = np.array([[float(item) for item in line.split()] for line in data])
    return matrix
def z_score_normalize(matrix):
  mean = np.mean(matrix)
  std = np.std(matrix)
  z_score_matrix = (matrix - mean) / std
  return z_score_matrix
def main():
  matrix = load_matrix("mi/m0.txt")
  z_score_matrix = z_score_normalize(matrix)
  x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
  plt.pcolormesh(x, y, z_score_matrix)
  plt.colorbar()
  plt.show()
if __name__ == '__main__':
  main()
