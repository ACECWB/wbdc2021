import torch
import sys
sys.path.append('..')
from evaluation import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score
from tqdm import tqdm

class Session(object):
    def __init__(self, model, verbose=1):
        self.ACTION = ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'favorite', 'follow']
        self.action_w = [4/13, 3/13, 2/13, 1/13, 1/13, 1/13, 1/13]
        self.gpus = [0, 1]
        self.device = 'cuda:0'
        self.model = model
        self.verbose = verbose
        self.weight = torch.tensor([1.6, 1.2, 0.8, 0.4, 0.4, 0.4, 0.4], dtype=torch.float32).to(model.device)
        
    def compile(self, lr, optimizer, loss=None, metrics=None):
        self.metrics_names = ['loss']
        self.optim = self._get_optim(optimizer, lr=lr)
        self.loss_func = self._get_loss(loss)
        self.metrics = self._get_metrics(metrics)
        
    def multi_loss(self, y_pred, y_label):
        loss = F.binary_cross_entropy(y_pred, y_label, reduction='none')
        weight = torch.unsqueeze(self.weight, dim=0)
        return torch.mean(loss * weight)

    def train(self, train_loader):
#         self.model.train()
        model = self.model.train()
        model = nn.DataParallel(self.model, device_ids=self.gpus)
    
        logs = {}
        total_loss_epoch = 0.0
        train_result = {}
        sample_num = len(train_loader)
        run_sample_num = [sample_num] * len(self.metrics)
        userlist = []
        y_labels = {}
        y_preds = {}
        for i in self.ACTION:
            y_labels[i] = []
            y_preds[i] = []
        try:
            with tqdm(enumerate(train_loader), disable=self.verbose != 1) as t:
                for _, (train_x, train_y) in t:
                    x = train_x.to(self.device).float()
#                     print(x.shape, x)
                    userlist += x[:, 0].cpu().data.numpy().tolist()
                    y = train_y.to(self.device).float()
                    y_pred = model(x).squeeze()
            
                    for i in range(len(self.ACTION)):
                        y_labels[self.ACTION[i]] += y[:, i].cpu().data.numpy().tolist()
                        y_preds[self.ACTION[i]] += y_pred[:, i].cpu().data.numpy().tolist()
                        
#                     print(y_pred.shape, y_pred)
#                     print(y.shape, y)
#                     print(y_pred.shape)
                    lls = self.loss_func(y_pred, y)
                    reg_ls = self.model.get_regularization_loss()
                    loss = lls + reg_ls + self.model.aux_loss
                    total_loss_epoch += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if self.verbose > 0:
                        for i, data in enumerate(self.metrics.items()):
                            name, metric_func = data[0], data[1]
                            if name not in train_result:
                                train_result[name] = []
                            try:
                                train_result[name].append(metric_func(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype('float32')))
                            except ValueError:
                                run_sample_num[i] = run_sample_num[i] - 1
                                continue
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        logs['loss'] = total_loss_epoch / sample_num
        for i, data in enumerate(train_result.items()):
            name, result = data[0], data[1]
            logs[name] = np.sum(result) / (sample_num - run_sample_num[i])
        r = {}
        r_uauc = 0
        for i in range(len(self.ACTION)):
            target = self.ACTION[i]
            r[target] = uAUC(y_labels[target], y_preds[target], userlist)
            r_uauc += r[target] * self.action_w[i]
        print(r)
        print('Train result uauc: ', r_uauc)
        return logs
    
    def evaluate(self, val_x, val_y, table, batch_size=1024):
        eval_result = {}
        val_tensor_data = self.model.generate_loader(val_x, None, table, None)
        val_loader = DataLoader(val_tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans, userlist = self.predict(val_loader, is_valid=True)
        
        y_preds = {}
        y_labels = {}
        r = {}
        r_uauc = 0.0
        for i in range(len(self.ACTION)):
            target = self.ACTION[i]
            y_labels[target] = val_y[:, i]
            y_preds[target] = pred_ans[:, i]
            r[target] = uAUC(y_labels[target], y_preds[target], userlist)
            r_uauc += r[target] * self.action_w[i]
        print(r)
        print('Valid result uauc: ', r_uauc)
        
#         for name, metric_func in self.metrics.items():
#             eval_result[name] = metric_func(pred_ans, val_y)
        return eval_result
        
    def predict(self, val_loader, is_valid=False):
        self.model.eval()
        pred_ans = []
        if not is_valid:
            with torch.no_grad():
                for t in val_loader:
                    x = t.to(self.model.device).float()
                    y_pred = self.model(x).cpu().data.numpy()
                    pred_ans.append(y_pred)
            return np.concatenate(pred_ans).astype('float64')
        else:
            y_preds = {}
            y_labels = {}
            userlist = []
            for i in self.ACTION:
                y_labels[i] = []
                y_preds[i] = []
                
            with torch.no_grad():
                for t in val_loader:
                    x = t.to(self.model.device).float()
                    userlist += x[:, 0].cpu().data.numpy().tolist()
                    y_pred = self.model(x).cpu().data.numpy()
#                     for i in range(len(self.ACTION)):
#                         y_labels[self.ACTION[i]] += y[:, i].cpu().data.numpy().tolist()
#                         y_preds[self.ACTION[i]] += y_pred[:, i].cpu().data.numpy(;).tolist()
    
                    pred_ans.append(y_pred)
            return np.concatenate(pred_ans).astype('float64'), userlist
            
    
    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_
        
    
    def _get_optim(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            lr = kwargs['lr']
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.model.parameters(), lr=lr)
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.model.parameters(), lr=lr)
            else:
                raise NotImplementedError

        else:
            optim = optimizer

        return optim
    def _get_loss(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "multi_binary_crossentroy":
                loss_func = self.multi_loss
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func
    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)