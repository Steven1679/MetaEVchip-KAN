import os
import torch
import pandas as pd
from torch.utils import data
import dataUnit


class DataSet(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.devices_path = list()
        self.devices = list()
        self.load_data()

    def load_data(self):
        file_path = os.path.join(self.args.ROOT_DIR, "dataset", "dynamic_monitoring.xlsx")
        try:
            excel_data = pd.read_excel(file_path, header=None)
            for col in range(excel_data.shape[1]):
                device = dataUnit.DataUnit(self.args, col, excel_data)
                self.devices.append(device)

        except Exception as e:
            print(f"Error loading data: {e}")

    def __getitem__(self, index):
        device = self.devices[index]
        shift = torch.tensor(device.shift)
        spectrum = torch.tensor(device.spectrum)
        return shift, spectrum

    def __len__(self):
        return len(self.devices)
