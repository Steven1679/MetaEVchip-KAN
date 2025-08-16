import argparse
import torch
import os


class MyParse(object):
    def __init__(self):
        self.__parser = argparse.ArgumentParser('KAN parse')
        self.__parser.add_argument('--ROOT_DIR', default=self.__get_root_dir(), help="the root dir of project")
        self.__parser.add_argument('--epochs', default=2000, type=int, help="total epoch needed to run")
        self.__parser.add_argument('--train_batch_size', default=400, type=int, help="batch size on training set")
        self.__parser.add_argument('--val_batch_size', default=100, type=int, help="batch size on validation set")
        self.__parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                   type=str, help="available device")
        self.__parser.add_argument('--lr', default=1e-3, help="learning rate")
        self.__parser.add_argument('--seed', default=42, help="random seed")
        # obtain namespace object
        self.args = self.__parser.parse_args()
        print("\033[31mcuda is available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0), "\n")

    @staticmethod
    def __get_root_dir():
        root_dir = os.getcwd()
        while False if any("dataset" in s for s in os.listdir(root_dir)) else True:
            root_dir = os.path.dirname(root_dir)
        return root_dir
