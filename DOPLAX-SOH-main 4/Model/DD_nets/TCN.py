import torch
import torch.nn as nn
import numpy as np
import os
from Model.utils.util import AverageMeter, eval_metrix
from utils.util import get_logger, write_to_file, write_to_json
from Model.utils.lr_schedulers import LR_Scheduler
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class TCN_Net(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=5, dropout=0.2):
        super(TCN_Net, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout),
                       CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x is [1, SeqLen, features]
        x = x.transpose(1, 2) # [1, features, SeqLen]
        y = self.network(x) # [1, out_channels, SeqLen]
        y = y.transpose(1, 2) # [1, SeqLen, out_channels]
        out = self.linear(y) # [1, SeqLen, 1]
        return out

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.generator = torch.Generator(device=device)
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        self.log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(self.log_dir)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_func = nn.MSELoss()
        
        self.train_loss_history = []
        self.valid_loss_history = []
        self.test_metrics_history = {'MAE': [], 'MAPE': [], 'MSE': [], 'RMSE': []}
        self.best_model = None
        self.best_epoch = None
        
    def predict(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)

    def Train(self, trainloader, validloader=None, testloader=None):
        min_valid_mse = np.inf
        early_stop = 0
        
        for e in range(1, self.args.epochs+1):
            early_stop += 1
            self.train()
            loss_meter = AverageMeter()
            for x1, _, y1, _ in trainloader:
                x1, y1 = x1.to(device), y1.to(device)
                if self.model is None:
                    self.m = x1.shape[2]
                    self.model = TCN_Net(input_size=self.m, num_channels=[32, 32, 32, 32, 32, 32], kernel_size=5).to(device)

                    base_lr = getattr(self.args, f'lr_{self.args.data}', getattr(self.args, 'lr', 0.01))
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr)
                    warmup_epochs = getattr(self.args, f'warmup_epochs_{self.args.data}', getattr(self.args, 'warmup_epochs', 30))
                    warmup_lr = getattr(self.args, f'warmup_lr_{self.args.data}', getattr(self.args, 'warmup_lr', 0.002))
                    final_lr = getattr(self.args, f'final_lr_{self.args.data}', getattr(self.args, 'final_lr', 1e-6))
                    self.scheduler = LR_Scheduler(
                        optimizer=self.optimizer,
                        warmup_epochs=warmup_epochs,
                        warmup_lr=warmup_lr,
                        num_epochs=getattr(self.args, 'epochs', 2000),
                        base_lr=base_lr,
                        final_lr=final_lr
                    )
                
                u1 = self.model(x1)
                loss = self.loss_func(u1, y1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item())


            if self.scheduler is not None:
                current_lr = self.scheduler.step()
            else:
                current_lr = getattr(self.args, 'lr', 0.01)
                
            self.train_loss_history.append(loss_meter.avg)
            info = f'[Train] epoch:{e}, lr:{current_lr:.6f}, total loss:{loss_meter.avg:.6f}'
            self.logger.info(info)

            if validloader is not None:
                valid_mse = self.Valid(validloader)
                self.valid_loss_history.append(valid_mse)
                self.logger.info(f'[Valid] epoch:{e}, MSE: {valid_mse}')
                
                if valid_mse < min_valid_mse and testloader is not None:
                    min_valid_mse = valid_mse
                    true_label, pred_label = self.Test(testloader)
                    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                    self.test_metrics_history['MAE'].append(MAE)
                    self.test_metrics_history['MAPE'].append(MAPE)
                    self.test_metrics_history['MSE'].append(MSE)
                    self.test_metrics_history['RMSE'].append(RMSE)
                    self.logger.info(f'[Test] MSE: {MSE:.8f}, MAE: {MAE:.6f}, MAPE: {MAPE:.6f}, RMSE: {RMSE:.6f}')
                    early_stop = 0
                    self.best_epoch = e
                    self.best_model = self.model.state_dict()
                    
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                self.logger.info(f'early stop at epoch {e}')
                break
                

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        if self.args.save_folder:
            import matplotlib.pyplot as plt
            import os

            
            best_model_path = os.path.join(self.args.save_folder, 'best_model.pth')
            torch.save(self.best_model, best_model_path)
            
            plt.figure()
            plt.plot(self.train_loss_history, label='Train Loss')
            plt.plot(self.valid_loss_history, label='Valid Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.args.save_folder, 'training_history.png'), bbox_inches='tight')
            plt.close()
            
            true_kpd, pred_kpd = self.Test(testloader)
            np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_kpd)
            np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_kpd)
            
        return 'valid', None


    def Valid(self, validloader):
        self.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(device)
                u1 = self.model(x1)
                true_label.append(y1.cpu().numpy().flatten())
                pred_label.append(u1.cpu().numpy().flatten())
        mse = self.loss_func(torch.tensor(np.concatenate(pred_label)), torch.tensor(np.concatenate(true_label)))
        return mse.item()

    def Test(self, testloader):
        self.eval()
        true_label, pred_label = [], []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(device)
                u1 = self.model(x1)
                true_label.append(y1.cpu().numpy().flatten())
                pred_label.append(u1.cpu().numpy().flatten())
        return np.concatenate(true_label), np.concatenate(pred_label)
