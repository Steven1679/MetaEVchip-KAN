import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

import dataSet
import parseUtils
from kan_inverse import InverseKAN

# file name
Structure_Title = "kan_inverse.pth"
CheckPoint = "checkpoints\\" + Structure_Title


def save_checkpoint(state, filename=CheckPoint):
    torch.save(state, filename)


def cal_accuracy(shift_out, shift_true):
    criterion = nn.MSELoss()
    shift_accuracy = criterion(shift_out, shift_true)
    return shift_accuracy


class Main(object):
    def __init__(self, args):
        self.args = args
        self.fix_seed()
        self.args.writer = SummaryWriter(log_dir='logs')
        self.dataset = dataSet.DataSet(self.args)

        # set kfold parameters
        self.k_folds = 5
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.args.seed)

    def train(self, load_checkpoint=False):
        # initialize storage list
        fold_val_losses = []
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.dataset)):
            print(f'==================== Fold {fold + 1}/{self.k_folds} ====================')

            # load dataset
            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            val_subset = torch.utils.data.Subset(self.dataset, val_idx)
            train_loader = DataLoader(dataset=train_subset, batch_size=self.args.train_batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_subset, batch_size=self.args.val_batch_size, shuffle=False)

            # initialize model
            model = InverseKAN(self.args).to(self.args.device)
            criterion = nn.MSELoss().to(self.args.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            start_epoch = 0

            # load model state
            if load_checkpoint:
                checkpoint_path = f"checkpoints/{Structure_Title}_fold{fold + 1}.pth"
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['state_dict'])

            # model training
            for epoch in range(start_epoch, self.args.epochs):
                model.train()
                train_loss = 0
                self.adjust_learning_rate(optimizer, epoch)
                for i, (shift_true, spectrum) in enumerate(train_loader):
                    shift_true = shift_true.to(self.args.device)
                    spectrum = spectrum.to(self.args.device)
                    outputs = model(spectrum)
                    loss = criterion(outputs, shift_true)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # model validation
                val_loss = self.validation(model, val_loader)

                print('\033[34mFold:', fold + 1, ' Epoch', '%04d' % epoch, ">" * 8,
                      'train_loss =', '{:.9f}'.format(train_loss), ">" * 8,
                      'val_loss =', '{:.9f}'.format(val_loss))
                self.visualization(train_loss, val_loss, epoch, fold)
            print(f'==================== Fold {fold + 1} Completed ====================')

        # average loss
        avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
        print(f'\n==================== Cross-Validation Completed ====================')
        print(f'Average Validation Loss: {avg_val_loss:.9f}')

    def validation(self, model, val_loader):
        model.eval()
        val_loss = 0.0
        sample_len = 0
        all_outputs = []
        all_shift = []

        with torch.no_grad():
            for i, (shift_true, spectrum) in enumerate(val_loader):
                shift_true = shift_true.to(self.args.device)
                spectrum = spectrum.to(self.args.device)
                outputs = model(spectrum)
                loss = cal_accuracy(outputs, shift_true)
                val_loss += loss.item()
                sample_len += shift_true.size(0)

                # result collection
                all_outputs.append(outputs)
                all_shift.append(shift_true)
        val_loss /= sample_len
        return val_loss

    def visualization(self, train_loss, val_loss, epoch, fold):
        self.args.writer.add_scalar(f'{Structure_Title}_Fold{fold + 1}_train_loss', train_loss, epoch)
        self.args.writer.add_scalar(f'{Structure_Title}_Fold{fold + 1}_val_loss', val_loss, epoch)

    def fix_seed(self):
        if self.args.seed is not None:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * np.power(0.95, epoch // 80)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    args = parseUtils.MyParse().args
    m = Main(args)
    m.train()
