import pandas as pd

class CaliforniaHousingDataset:
    path = "datasets/housing.csv"
    
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop(columns=['median_house_value']).values
        self.targets = self.data['median_house_value'].values.reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]