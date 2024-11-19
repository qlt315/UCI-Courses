import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd

class UCIDataLoader:
    def __init__(self,dataset_id):
        # fetch dataset
        adult = fetch_ucirepo(id=dataset_id)
        # data (as pandas dataframes)
        self.X = adult.data.features
        self.y = adult.data.targets

    def check_data_info(self):
        # Check basic information of feature data
        print("Feature Data Info:")
        print(self.X.info())
        print("\nTarget Data Info:")
        print(self.y.info())

        # Describe the dataset to get basic statistics
        print("\nFeature Data Description:")
        print(self.X.describe(include='all'))

    def clean_data(self):
        # Identify missing values represented as '?'
        missing_symbol = '?'
        missing_counts = (self.X == missing_symbol).sum()
        print("\n Missing Values Represented by '?':")
        print(missing_counts[missing_counts > 0])

        print("--------------------------------------------")
        missing_candidates = self.X.isin(['?', '', 'nan', 'N/A', None]).sum(axis=0)
        print("others?\n", missing_candidates[missing_candidates > 0])

        # Replace '?' with np.nan for standard missing value handling
        self.X = self.X.replace(missing_symbol, np.nan)

        print("--------------------------------------------")
        missing_candidates = self.X.isin(['?', '', 'nan', 'N/A', None]).sum(axis=0)
        print("others? after\n", missing_candidates[missing_candidates > 0])







if __name__ == "__main__":
    dataset_id = 2
    dataset = UCIDataLoader(dataset_id)
    

