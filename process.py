import numpy as np
import pandas as pd

def read_lncrna_indices(lncRNAid_file):
    lncRNA_ids = pd.read_csv(lncRNAid_file, header=None, names=['Index','name'])
    lncRNA_index = {index: idx for idx, index in enumerate(lncRNA_ids['Index'].values)}
    return lncRNA_index

def create_index_mapping(id_path):
    with open(id_path, 'r') as file:
        ids = [line.strip() for line in file]
    id_to_index = {int(id): idx for idx, id in enumerate(ids)}
    return id_to_index
def create_correlation_matrix(df, lncRNA_id_to_index, miRNA_id_to_index):
    num_rows = len(lncRNA_id_to_index)
    num_cols = len(miRNA_id_to_index)
    adj_matrix = np.zeros((num_rows, num_cols), dtype=int)
    for _, row in df.iterrows():
        row_idx = lncRNA_id_to_index[int(row[0])]
        col_idx = miRNA_id_to_index[int(row[1])]
        adj_matrix[row_idx, col_idx] = 1
    return adj_matrix

def GIP_kernel(feature_matrix):
    nc = feature_matrix.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(feature_matrix)  
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(feature_matrix[i, :] - feature_matrix[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.exp(-temp_up / r)
    return matrix

def getGosiR(feature_matrix):
    nc = feature_matrix.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(feature_matrix[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r


