import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class OpenBHB_Dataset:
    def __init__(self, train_csv_path, val_csv_path, test_csv_path):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path

    def read_csv_data(self, file_path):
        try:
            data = pd.read_csv(file_path, delimiter=";")
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        return data

    def process_data(self, data, subset):
        data = data[['ID', 'Scanner', 'Gender', 'Age']].copy()
        data.loc[:, 'Loc'] = data['ID'].apply(lambda id: f"OpenBHB/{subset}_quasiraw/sub-{int(id)}_preproc-quasiraw_T1w.npy")
        data.loc[data['Scanner'] == 2, 'Scanner'] = 71
        data.loc[data['Scanner'] == 1, 'Scanner'] = 72
        data = data.dropna(subset=['Loc', 'Age', 'Gender'])
        data['patient_id'] = 'ID_0'
        return data

    def get_data(self):
        train_data = self.read_csv_data(self.train_csv_path)
        train_data = self.process_data(train_data, 'train')

        val_data = self.read_csv_data(self.val_csv_path)
        val_data = self.process_data(val_data, 'val')

        test_data = self.read_csv_data(self.test_csv_path)
        test_data = self.process_data(test_data, 'test')
        
        combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

        train_data, test_data = self.split_test_set(combined_data)

        return train_data, test_data

    def split_test_set(self, data):
        age_bins = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80)]
        test_set = pd.DataFrame()
        samples_per_bin = 3

        for lower, upper in age_bins:
            bin_data = data[(data['Age'] >= lower) & (data['Age'] < upper)]
            if len(bin_data) >= samples_per_bin:
                test_bin = bin_data.sample(n=samples_per_bin, random_state=42)
                data = data.drop(test_bin.index)
                test_set = pd.concat([test_set, test_bin])
            else:
                raise ValueError(f"Not enough samples in age bin {lower}-{upper}")

        return data, test_set
        