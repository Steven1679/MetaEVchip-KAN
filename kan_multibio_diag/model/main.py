import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import shap
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score

import parseUtils
from kan_inverse import InverseKAN

# file name
Structure_Title = "kan_inverse.pth"
CheckPoint = "checkpoints\\" + Structure_Title
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("images/shap_values"):
    os.makedirs("images/shap_values")


def save_checkpoint(state, filename=CheckPoint):
    torch.save(state, filename)


def cal_accuracy(health_status_out, health_status_true):
    criterion = nn.BCELoss()
    health_status_accuracy = criterion(health_status_out, health_status_true)
    return health_status_accuracy


def confusion_map(outputs, health_status, epoch, save_path):
    y_true = health_status.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    threshold = 0.5
    y_pred_dis = np.where(y_pred > threshold, 1, 0)
    conf_matrix = confusion_matrix(y_true, y_pred_dis)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("Predicted health status", fontsize=20)
    plt.ylabel("Actual health status", fontsize=20)
    plt.title("Confusion Matrix - Epoch {}".format(epoch), fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(save_path, dpi=500)
    plt.close()
    return conf_matrix


def roc_plot(outputs, health_status, epoch, save_path):
    fpr, tpr, _ = roc_curve(health_status.cpu().numpy(), outputs.cpu().numpy())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curve - Epoch {}'.format(epoch), fontsize=20)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(save_path, dpi=500)
    plt.close()
    return fpr, tpr, roc_auc


def save_auc_to_excel(auc_data, excel_writer, sheet_name):
    df_auc = pd.DataFrame(auc_data)
    df_auc.to_excel(excel_writer, sheet_name=sheet_name, index=False)


def save_accuracy_to_excel(accuracy_data, excel_writer, sheet_name):
    df_accuracy = pd.DataFrame(accuracy_data)
    df_accuracy.to_excel(excel_writer, sheet_name=sheet_name, index=False)


def save_confusion_matrix_to_excel(confusion_matrix_data, excel_writer, sheet_name):
    df_cm = pd.DataFrame(confusion_matrix_data, columns=["Phase", "Confusion Matrix"])
    df_cm.to_excel(excel_writer, sheet_name=sheet_name, index=False)


def compute_shap_values(model, X_train, device, feature_names, save_path):
    model.eval()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    explainer = shap.GradientExplainer(model, X_train_tensor)
    shap_values = explainer.shap_values(X_train_tensor)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values).squeeze()
    else:
        shap_values = shap_values.squeeze()
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(-1, 1)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False, plot_type="bar")
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    return shap_values


def save_shap_to_excel(shap_values, feature_names, filename="shap_data.xlsx"):
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    df_shap.to_excel(filename, index=False)


class Main(object):
    def __init__(self, args):
        self.args = args
        self.fix_seed()
        self.all_feature_combinations = []
        self.prepare_data()
        self.args.writer = SummaryWriter(log_dir='logs')
        self.feature_combination_dict = {}
        self.global_best_auc = 0
        self.global_best_model = None
        self.global_best_features = None
        self.global_best_scaler = None

    def prepare_data(self):
        self.train_val_data = pd.read_excel('dataset.xlsx')
        self.test_data = pd.read_excel('test.xlsx')

        # feature combination iteration
        features = self.train_val_data.iloc[:, 1:]
        self.all_feature_combinations = []
        for r in range(1, len(features.columns) + 1):
            self.all_feature_combinations.extend(combinations(features.columns, r))
        self.feature_combination_dirs = {}
        for combo in self.all_feature_combinations:
            combo_name = '_'.join(combo)
            dir_path = os.path.join("results", combo_name)
            os.makedirs(dir_path, exist_ok=True)
            self.feature_combination_dirs[combo] = dir_path
        self.feature_combination_dict = {combo: idx + 1 for idx, combo in enumerate(self.all_feature_combinations)}

    def train(self):
        with pd.ExcelWriter('auc_data.xlsx') as auc_writer, \
                pd.ExcelWriter('accuracy_data.xlsx') as accuracy_writer, \
                pd.ExcelWriter('confusion_matrix_data.xlsx') as cm_writer, \
                pd.ExcelWriter('cv_results.xlsx') as cv_writer:

            # feature combination iteration
            for combo in self.all_feature_combinations:
                selected_features = list(combo)
                print(f"Training with features: {selected_features}")

                # create directory
                combo_dir = self.feature_combination_dirs.get(tuple(selected_features))
                if combo_dir is None:
                    combo_name = '_'.join(selected_features)
                    combo_dir = os.path.join("results", combo_name)
                    os.makedirs(combo_dir, exist_ok=True)
                    self.feature_combination_dirs[tuple(selected_features)] = combo_dir
                cm_dir = os.path.join(combo_dir, "confusion_matrix")
                roc_dir = os.path.join(combo_dir, "roc_curves")
                shap_dir = os.path.join(combo_dir, "shap_values")
                cv_dir = os.path.join(combo_dir, "cv_results")
                os.makedirs(cm_dir, exist_ok=True)
                os.makedirs(roc_dir, exist_ok=True)
                os.makedirs(shap_dir, exist_ok=True)
                os.makedirs(cv_dir, exist_ok=True)

                # data prepare
                train_val_data = self.train_val_data[['cancer_result'] + selected_features]
                test_data = self.test_data[['cancer_result'] + selected_features]
                X = train_val_data.iloc[:, 1:].values.astype(np.float32)
                y = train_val_data.iloc[:, 0].values.astype(np.float32)

                # k fold
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_results = []
                best_models = []  # best model

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    print(f"  Fold {fold + 1}/5")
                    auc_data = []
                    accuracy_data = []
                    confusion_matrix_data = []

                    # divide data
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_val_fold = scaler.transform(X_val_fold)

                    # load data
                    X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1)
                    X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).view(-1, 1)
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
                    val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=False)

                    # initialize model
                    model = InverseKAN(input_size=len(selected_features)).to(self.args.device)
                    criterion = nn.BCELoss().to(self.args.device)
                    optimizer = optim.Adam(model.parameters(), lr=1e-4)

                    # model training of current fold
                    best_val_auc = 0
                    best_model_state = None
                    for epoch in range(self.args.epochs):
                        model.train()
                        self.adjust_learning_rate(optimizer, epoch)
                        train_loss = 0
                        for i, (spectrum, health_status_true) in enumerate(train_loader):
                            spectrum, health_status_true = spectrum.to(self.args.device), health_status_true.to(
                                self.args.device)
                            outputs = model(spectrum)
                            loss = criterion(outputs, health_status_true)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                        train_loss /= len(train_loader)
                        val_loss, val_auc, val_accuracy = self.validation(
                            model, val_loader, epoch,
                            os.path.join(cm_dir, f"fold{fold + 1}_confusion_matrix_val_epoch{epoch}.svg"),
                            os.path.join(roc_dir, f"fold{fold + 1}_roc_curve_val_epoch{epoch}.svg"),
                            auc_data, accuracy_data, confusion_matrix_data
                        )

                        # update best model
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_model_state = model.state_dict()
                        print(f'\033[34mFold {fold + 1} Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_auc={val_auc:.4f}')

                    # save best model of current fold
                    best_models.append({
                        'state_dict': best_model_state,
                        'val_auc': best_val_auc,
                        'scaler': scaler
                    })
                    fold_results.append({
                        'fold': fold + 1,
                        'best_val_auc': best_val_auc
                    })

                # kfold result storage
                cv_df = pd.DataFrame(fold_results)
                cv_df.to_excel(os.path.join(cv_dir, "cross_validation_results.xlsx"), index=False)
                avg_val_auc = cv_df['best_val_auc'].mean()
                print(f"\033[32mAverage Validation AUC for feature combo {selected_features}: {avg_val_auc:.4f}")

                # select best model for testing
                best_fold_idx = np.argmax([model['val_auc'] for model in best_models])
                best_model_info = best_models[best_fold_idx]
                print(
                    f"Selected best model from fold {best_fold_idx + 1} with val_auc={best_model_info['val_auc']:.4f}")

                # initialize model
                best_model = InverseKAN(input_size=len(selected_features)).to(self.args.device)
                best_model.load_state_dict(best_model_info['state_dict'])

                X_test = test_data.iloc[:, 1:].values.astype(np.float32)
                y_test = test_data.iloc[:, 0].values.astype(np.float32)
                scaler = best_model_info['scaler']
                X_test_scaled = scaler.transform(X_test)
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.test_batch_size, shuffle=False)

                # test
                test_loss, test_auc, test_accuracy, test_cm, test_fpr, test_tpr = self.test(
                    best_model, test_loader, self.args.epochs,
                    os.path.join(cm_dir, "confusion_matrix_test.svg"),
                    os.path.join(roc_dir, "roc_curve_test.svg")
                )

                print(f'\033[32mTesting complete for feature combo {selected_features}: test_loss={test_loss:.6f}, test_auc={test_auc:.4f}')

                # update global best model
                if test_auc > self.global_best_auc:
                    self.global_best_auc = test_auc
                    self.global_best_model = best_model
                    self.global_best_features = selected_features
                    self.global_best_scaler = scaler
                    print(f"\033[35mNew global best model! Features: {selected_features}, Test AUC: {test_auc:.4f}")

                # save test results
                test_auc_data = [{"Epoch/Phase": "test", "FPR": test_fpr, "TPR": test_tpr, "AUC": test_auc}]
                test_accuracy_data = [{"Epoch/Phase": "test", "Accuracy": test_accuracy, "AUC": test_auc}]
                test_confusion_matrix_data = [("test", test_cm)]
                combo_id = self.feature_combination_dict.get(tuple(selected_features))
                if combo_id is None:
                    combo_id = len(self.feature_combination_dict) + 1
                    self.feature_combination_dict[tuple(selected_features)] = combo_id
                if test_auc_data:
                    save_auc_to_excel(test_auc_data, auc_writer, f"combo_{combo_id}")
                if test_accuracy_data:
                    save_accuracy_to_excel(test_accuracy_data, accuracy_writer, f"combo_{combo_id}")
                if test_confusion_matrix_data:
                    save_confusion_matrix_to_excel(test_confusion_matrix_data, cm_writer, f"combo_{combo_id}")
                cv_df.to_excel(cv_writer, sheet_name=f"combo_{combo_id}_cv", index=False)

                # SHAP analyze
                shap_values = compute_shap_values(
                    best_model, X_train_fold, self.args.device, selected_features,
                    os.path.join(shap_dir, "shap_summary_plot.svg")
                )
                save_shap_to_excel(
                    shap_values, selected_features,
                    filename=os.path.join(shap_dir, "shap_data.xlsx")
                )

            # save global best model
            if self.global_best_model:
                print(f"\n\033[35mGlobal Best Model Features: {self.global_best_features}")
                print(f"Global Best Test AUC: {self.global_best_auc:.4f}")
                torch.save({
                    'state_dict': self.global_best_model.state_dict(),
                    'features': self.global_best_features,
                    'test_auc': self.global_best_auc
                }, 'best_model.pth')
                import joblib
                joblib.dump(self.global_best_scaler, 'best_scaler.pkl')
            self.save_feature_combination_map()

    def validation(self, model, val_loader, epoch, cm_save_path, roc_save_path, auc_data, accuracy_data, confusion_matrix_data):
        model.eval()
        with torch.no_grad():
            loss = 0
            sample_len = 0
            all_outputs = []
            all_health_status = []
            for i, (spectrum, health_status_true) in enumerate(val_loader):
                spectrum, health_status_true = spectrum.to(self.args.device), health_status_true.to(self.args.device)
                outputs = model(spectrum)
                loss += cal_accuracy(outputs, health_status_true)
                sample_len += health_status_true.shape[0]
                all_outputs.extend(outputs.cpu().numpy())
                all_health_status.extend(health_status_true.cpu().numpy())

            # model evaluation
            cm = confusion_map(
                torch.tensor(all_outputs), torch.tensor(all_health_status),
                epoch, save_path=cm_save_path
            )
            confusion_matrix_data.append((epoch, cm))
            fpr, tpr, roc_auc = roc_plot(
                torch.tensor(all_outputs), torch.tensor(all_health_status),
                epoch, save_path=roc_save_path
            )
            auc_data.append({"Epoch/Phase": epoch, "FPR": fpr, "TPR": tpr, "AUC": roc_auc})
            threshold = 0.5
            y_pred = np.where(np.array(all_outputs) > threshold, 1, 0)
            y_true = all_health_status
            accuracy = accuracy_score(y_true, y_pred)
            accuracy_data.append({"Epoch/Phase": epoch, "Accuracy": accuracy, "AUC": roc_auc})
            loss /= sample_len
        return loss, roc_auc, accuracy

    def test(self, model, test_loader, epoch, cm_save_path, roc_save_path):
        model.eval()
        with torch.no_grad():
            loss = 0
            sample_len = 0
            all_outputs = []
            all_health_status = []
            for i, (spectrum, health_status_true) in enumerate(test_loader):
                spectrum, health_status_true = spectrum.to(self.args.device), health_status_true.to(self.args.device)
                outputs = model(spectrum)
                loss += cal_accuracy(outputs, health_status_true)
                sample_len += health_status_true.shape[0]
                all_outputs.extend(outputs.cpu().numpy())
                all_health_status.extend(health_status_true.cpu().numpy())

            # model evaluation
            cm = confusion_map(
                torch.tensor(all_outputs), torch.tensor(all_health_status),
                epoch, save_path=cm_save_path
            )
            fpr, tpr, roc_auc = roc_plot(
                torch.tensor(all_outputs), torch.tensor(all_health_status),
                epoch, save_path=roc_save_path
            )
            threshold = 0.5
            y_pred = np.where(np.array(all_outputs) > threshold, 1, 0)
            y_true = all_health_status
            accuracy = accuracy_score(y_true, y_pred)
            loss /= sample_len
        return loss, roc_auc, accuracy, cm, fpr, tpr

    def visualization(self, train_loss, val_loss, epoch):
        self.args.writer.add_scalar(Structure_Title + 'train_loss', train_loss, epoch)
        self.args.writer.add_scalar(Structure_Title + 'val_loss', val_loss, epoch)

    def fix_seed(self):
        if self.args.seed is not None:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.args.lr * np.power(0.9, epoch // 120)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_feature_combination_map(self):
        df_map = pd.DataFrame(list(self.feature_combination_dict.items()), columns=['feature_combination', 'ID'])
        df_map.to_excel('feature_combination_map.xlsx', index=False)


if __name__ == '__main__':
    args = parseUtils.MyParse().args
    m = Main(args)
    m.train()
