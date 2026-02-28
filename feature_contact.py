import pandas as pd
import numpy as np

data_path = '../data/nuswidelite/NUS-WIDE-Lite/NUS-WIDE-Lite_features'

file_color_histogram = data_path + '/' + 'Normalized_CH_Lite_Train.dat'
file_edge_direction_histogram = data_path + '/' + 'Normalized_EDH_Lite_Train.dat'

color_matrix = np.loadtxt(file_color_histogram)
edge_direction_matrix = np.loadtxt(file_edge_direction_histogram)

# contact
test_matrix = np.concatenate((color_matrix, edge_direction_matrix), axis=1)


# Re_Nomalized
mean = np.mean(test_matrix, axis=0)
std = np.std(test_matrix, axis=0)
test_matrix_line = (test_matrix - mean) / (3 * std)
norms = np.linalg.norm(test_matrix_line, axis=1, keepdims=True)
test_matrix_normalized = test_matrix_line / norms

print(test_matrix_normalized.shape)

## test
# for i in range(5):
#     squared_sum = np.sum(test_matrix_normalized[i] ** 2)
#     print(squared_sum)

file_test_histogram = data_path + '/' + 'Normalized_Lite_Train.dat'

np.savetxt(file_test_histogram, test_matrix_normalized)