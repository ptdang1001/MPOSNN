from torch.utils.data import Dataset
import numpy as np
import pandas as pd



class DataSet(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = self.data[index]
        y = self.labels[index]

        return X, y

def shuffle_geneExpression(geneExpression, by="full_random"):
    n_row, n_col = geneExpression.shape
    geneExpression_shuffled=None
    if by == "full_random":
        n_total = n_row * n_col
        tmp_values = geneExpression.values
        tmp_values = tmp_values.reshape(n_total)
        rdm_idxs = np.random.choice(n_total, n_total, replace=False)
        tmp_values = tmp_values[rdm_idxs]
        tmp_values = tmp_values.reshape(n_row, n_col)
        tmp_values = pd.DataFrame(tmp_values)
        tmp_values.index = geneExpression.index
        tmp_values.columns = geneExpression.columns
        geneExpression_shuffled = tmp_values

    return geneExpression_shuffled
