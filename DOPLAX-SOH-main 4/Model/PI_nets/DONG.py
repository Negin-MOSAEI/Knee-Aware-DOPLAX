import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import get_logger, write_to_file, write_to_json
import deepxde as dde
from Model.Backbones import Autoformer, CPMLP, DeepONet
from Model.utils.util import find_input_dim_from_checkpoint, AverageMeter, eval_metrix
from Model.Auxiliary_nets.Solution_u import Solution_u
from Model.Auxiliary_nets.MLP import MLP
from Model.utils.lr_schedulers import LR_Scheduler
from Model.utils.activation_functions import Sin
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import regex as re
import os
import warnings
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'




class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()


    def forward(self, x):
        return torch.sin(x)



class MLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, dropout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*self.layers)
        self._init()


    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)


    def forward(self,x):
        x = self.net(x)
        return x


                
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0


    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr


    def get_lr(self):
        return self.current_lr



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if self.args is None:
            return 
        
        # Create CUDA generator for DataLoaders
        self.generator = torch.Generator(device=device)
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        self.log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(self.log_dir)
        
        if self.args.data == "XJTU":
            self.args.lr = self.args.lr_XJTU
            self.args.warmup_lr = self.args.warmup_lr_XJTU
            self.args.lr_F = self.args.lr_F_XJTU
            self.args.final_lr = self.args.final_lr_XJTU
            self.args.F_hidden_dim = self.args.F_hidden_dim_XJTU
            self.args.F_layers_num = self.args.F_layers_num_XJTU
            self.args.dropout = self.args.dropout_XJTU
            self.args.alpha = self.args.alpha_XJTU
            self.args.beta = self.args.beta_XJTU
            self.args.warmup_epochs = self.args.warmup_epochs_XJTU
        elif self.args.data == "TJU":
            self.args.lr = self.args.lr_TJU
            self.args.warmup_lr = self.args.warmup_lr_TJU
            self.args.lr_F = self.args.lr_F_TJU
            self.args.final_lr = self.args.final_lr_TJU
            self.args.F_hidden_dim = self.args.F_hidden_dim_TJU
            self.args.F_layers_num = self.args.F_layers_num_TJU
            self.args.dropout = self.args.dropout_TJU
            self.args.alpha = self.args.alpha_TJU
            self.args.beta = self.args.beta_TJU
            self.args.warmup_epochs = self.args.warmup_epochs_TJU
        elif self.args.data == "MIT":
            self.args.lr = self.args.lr_MIT
            self.args.warmup_lr = self.args.warmup_lr_MIT
            self.args.lr_F = self.args.lr_F_MIT
            self.args.final_lr = self.args.final_lr_MIT
            self.args.F_hidden_dim = self.args.F_hidden_dim_MIT
            self.args.F_layers_num = self.args.F_layers_num_MIT
            self.args.dropout = self.args.dropout_MIT
            self.args.alpha = self.args.alpha_MIT
            self.args.beta = self.args.beta_MIT
            self.args.warmup_epochs = self.args.warmup_epochs_MIT
        else:
            self.args.lr = self.args.lr_HUST
            self.args.warmup_lr = self.args.warmup_lr_HUST
            self.args.lr_F = self.args.lr_F_HUST
            self.args.final_lr = self.args.final_lr_HUST
            self.args.F_hidden_dim = self.args.F_hidden_dim_HUST
            self.args.F_layers_num = self.args.F_layers_num_HUST
            self.args.dropout = self.args.dropout_HUST
            self.args.alpha = self.args.alpha_HUST
            self.args.beta = self.args.beta_HUST
            self.args.warmup_epochs = self.args.warmup_epochs_HUST
            
        self._save_args()

        # Act as F
        self.deeponet = None
        self.optimizer_deeponet = None
        
        # Act as G
        if self.args.F_hidden_dim <= 20:
            hidden_dim_F = self.args.F_hidden_dim
        else:
            hidden_dim_F = self.args.F_hidden_dim*2
        self.dynamical_F = MLP(input_dim=35,output_dim=1,
                               layers_num=self.args.F_layers_num, 
                               hidden_dim=hidden_dim_F,
                               dropout=self.args.dropout).to(device)
        self.optimizer_F = torch.optim.Adam(self.dynamical_F.parameters(), lr=self.args.lr_F)



        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # loss history tracking
        self.train_loss_history = []
        self.valid_loss_history = []
        self.test_metrics_history = {
            'MAE': [],
            'MAPE': [],
            'MSE': [],
            'RMSE': []
        }
        
        # the best model
        self.best_model = None
        self.best_epoch = None


    def _save_args(self):
        """
        Save the parameters in parser to self.logger
        """
        if self.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    
    def _save_model(self):
        if self.args.save_folder is not None:
            best_model_path = os.path.join(self.args.save_folder, f'best_model.pth')
            torch.save(self.best_model, best_model_path)
            write_to_file(file_path=self.log_dir, info=f'The best model created at epoch {self.best_epoch}')
            

    def _clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()


    def _freeze_weights(self, freeze=True):
        """
        Freeze/unfreeze layer weights
        """
        for param in self.deeponet.parameters():
            param.requires_grad = not freeze

        
    def load_model(self, model_path, freeze):
        # Load checkpoint to CPU first
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize deeponet if it doesn't exist
        if self.deeponet is None:
            try:
                self.m = checkpoint['input_dimention']
            except:
                self.m = find_input_dim_from_checkpoint(checkpoint_path=model_path)
                
            layer_sizes_branch = [self.m] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
            layer_sizes_trunk = [self.args.dim_x] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
            self.deeponet = dde.nn.DeepONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation=Sin(),
                kernel_initializer="Glorot normal"
            ).to(device)
        
        self.deeponet.load_state_dict(checkpoint['deeponet'], strict=False)
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        
        self._freeze_weights(freeze=freeze)


    def predict(self, xt):
        coordinates = torch.linspace(0, 1, self.m-1)
        u_inputs = (xt, coordinates[-1].reshape(-1, 1).to(device))
        encoded_u_inputs = (xt, coordinates.reshape(-1, 1).to(device))
        return self.deeponet(encoded_u_inputs), self.deeponet(u_inputs)
             
            
    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []

        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                _, u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)

        return true_label,pred_label


    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []

        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                _, u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()


    def forward(self, xt):
        xt.requires_grad = True

        x = xt[:, :-1]
        t = xt[:, -1:]
        xt = torch.cat((x,t), dim=1)
        
        encoded_u, u = self.predict(xt)

        u_t = grad(u.sum(), t, 
                   create_graph=True, 
                   only_inputs=True, 
                   allow_unused=True)[0]
        u_x = grad(u.sum(), x, 
                   create_graph=True, 
                   only_inputs=True, 
                   allow_unused=True)[0]
        
        F = self.dynamical_F(torch.cat([xt, u, u_x, u_t], dim=1))
        f = u_t - F
    
        return u, f


    def train_one_epoch(self, epoch, dataloader):
        self.train()

        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            if self.deeponet is None:
                self.m = x1.shape[1]
                layer_sizes_branch = [self.m] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
                layer_sizes_trunk = [self.args.dim_x] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
                self.deeponet = dde.nn.DeepONetCartesianProd( 
                    layer_sizes_branch=layer_sizes_branch, 
            	    layer_sizes_trunk=layer_sizes_trunk,      
            	    activation=Sin(),
            	    kernel_initializer="Glorot normal"
            	).to(device)
                self.optimizer_deeponet = torch.optim.Adam(self.deeponet.parameters(), lr=self.args.warmup_lr)
                
                self.scheduler = LR_Scheduler(
                    optimizer=self.optimizer_deeponet, 
                    warmup_epochs=self.args.warmup_epochs,
                    warmup_lr=self.args.warmup_lr,
                    num_epochs=self.args.epochs,
                    base_lr=self.args.lr,
                    final_lr=self.args.final_lr
                )
        
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            
            # data loss
            loss1 = 0.5*self.loss_func(u1, y1) + 0.5*self.loss_func(u2, y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1, f_target) + 0.5*self.loss_func(f2, f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1, y1-y2)).sum()

            # total loss
            loss = loss1 + self.args.alpha*loss2 + self.args.beta*loss3

            self.optimizer_deeponet.zero_grad()
            self.optimizer_F.zero_grad()
            
            loss.backward()
            
            self.optimizer_deeponet.step()
            self.optimizer_F.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())

            # Logging
            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch, iter+1, loss1, loss2, loss3))

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg


    def Train(self, trainloader, validloader=None, testloader=None):
        if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'generator'):
            trainloader.sampler.generator = self.generator

        if hasattr(validloader, 'sampler') and hasattr(validloader.sampler, 'generator'):
            validloader.sampler.generator = self.generator
            
        # min_valid_mse = 10
        min_valid_mse = np.inf
        valid_mse = 10
        early_stop = 0

        for e in range(1, self.args.epochs+1):
            early_stop += 1
            loss1, loss2, loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            total_loss = loss1+ self.args.alpha*loss2 + self.args.beta*loss3
            self.train_loss_history.append(total_loss)
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, current_lr, total_loss)
            self.logger.info(info)

            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                self.valid_loss_history.append(valid_mse)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
                self.logger.info(info)

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                self.test_metrics_history['MAE'].append(MAE)
                self.test_metrics_history['MAPE'].append(MAPE)
                self.test_metrics_history['MSE'].append(MSE)
                self.test_metrics_history['RMSE'].append(RMSE)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                self.best_epoch = e
                self.best_model = {
                    'input_dimention': self.m,
                    'deeponet': self.deeponet.state_dict(),
                    'dynamical_F': self.dynamical_F.state_dict()
                }
                
                if self.args.save_folder is not None:
                    y_true_path = os.path.join(self.args.save_folder, 'true_label.npy')
                    y_pred_path = os.path.join(self.args.save_folder, 'pred_label.npy')
                    np.save(y_true_path, true_label)
                    np.save(y_pred_path, pred_label)

            if self.args.early_stop is not None and early_stop > self.args.early_stop//4:
                if self.test_metrics_history['RMSE'][-1] > 0.50 or self.test_metrics_history['MAPE'][-1] > 0.50:
                    self._clear_logger()
                    return 'invalid', (self.test_metrics_history['RMSE'][-1], self.test_metrics_history['MAPE'][-1])
            
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break

        self._save_loss_history_plot()
        if self.args.run_optuna or self.args.run_samll_sample:
            pass
        else:
            self._save_differences_plot(y_true_path=y_true_path, y_pred_path=y_pred_path)
        self._save_model()
        self._clear_logger()
        return 'valid', None


    def _save_loss_history_plot(self):
        """
        Save the graph of the loss trajectory during training
        """
        if not (self.train_loss_history and self.valid_loss_history):
            print("No loss history to plot")
            return
            
        plt.figure(figsize=(12, 10))
        
        # Plot training and validation loss
        epochs = range(1, len(self.train_loss_history) + 1)
        plt.semilogy(epochs, self.train_loss_history, 'gray', linewidth=2, label='Train Loss')
        plt.semilogy(epochs, self.valid_loss_history, 'r-', alpha=0.3, linewidth=2, label='Valid Loss')
        
        # Plot test metrics at their actual evaluation points
        colors = ['g', 'm', 'c', 'y']
        marker_styles = ['o', 's', 'D', '^']  # Different markers for each metric
        
        for i, (metric, values) in enumerate(self.test_metrics_history.items()):
            if values:
                # Create list of epochs where this metric was actually calculated
                eval_epochs = [epoch for epoch, val in zip(epochs, values) if val is not None]
                eval_values = [val for val in values if val is not None]
                
                if eval_values:
                    plt.semilogy(
                        eval_epochs,
                        eval_values,
                        f'{colors[i % len(colors)]}:',  # Dotted line
                        marker=marker_styles[i % len(marker_styles)],  # Add marker
                        markersize=0.5,
                        linewidth=1.5,
                        label=f'Test {metric}'
                    )
        
        plt.title(f"Training History (Best Epoch: {self.best_epoch})", pad=20)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel('Loss/Metric Value', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.subplots_adjust()
        
        # Save the plot
        if self.args.save_folder:
            plot_path = os.path.join(self.args.save_folder, 'training_history.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Saved training history plot to {plot_path}")
        else:
            plt.tight_layout()
            plt.show()

    
    def _save_differences_plot(self, y_true_path, y_pred_path):
        if "XJTU" in y_true_path or "TJU" in y_true_path:
            parts = os.path.normpath(y_true_path).split(os.sep)
            batch_num, exp_num = parts[-3], parts[-2]

            batch_num, exp_num = int(batch_num.split('-')[0]), int(exp_num.split('Experiment')[1])
        else:
            batch_num, exp_num = "one_batch", y_true_path.split('/')[-2]
            exp_num = int(exp_num.split('Experiment')[1])
            
        if "XJTU" in y_true_path:
            dataset_name = "XJTU"
        elif "TJU" in y_true_path:
            dataset_name = "TJU"
        elif "MIT" in y_true_path:
            dataset_name = "MIT"
        else:
            dataset_name = "HUST"

        try:
            y_true = np.load(y_true_path) 
            y_pred = np.load(y_pred_path) 
        except:
            print
            return

        if len(y_true[0]) == 1:
            cycle_ids = [1]
        else:
            cycles_num = y_true.shape[0]
            cycle_ids = np.random.choice(cycles_num, 25, replace=False)
            info = {
                'dataset_name': dataset_name,
                'batch_num': batch_num,
                'selected_cycles': cycle_ids.tolist(),
            }
            cycles_root = os.path.join(os.path.dirname(y_true_path), 'plots')
            if not os.path.exists(cycles_root):
                os.mkdir(cycles_root)
            cycles_path = os.path.join(cycles_root, 'selected_cycles_info.json')
            write_to_json(cycles_path, info)

        for i, cycle_id in enumerate(cycle_ids):
            plt.figure(figsize=(10, 5))

            if y_true.shape[1] == 1:
                y_true_cp = y_true.copy()
                y_true_cp = y_pred.copy()
                y_true_cp = y_true_cp.reshape(-1)
                y_true_cp = y_true_cp.reshape(-1)
            else:
                y_true_cp = y_true

            if len(y_true_cp.shape) == 1:
                actuals = y_true
                preds = y_pred
                if batch_num == "one_batch":
                    title = f"SOH Degradation Plot - (Dataset:{dataset_name}-Experiment:{exp_num})"
                    file_name = f"differences_plot_experiment{exp_num}"
                else:
                    title = f"SOH Degradation Plot - (Dataset:{dataset_name}-Batch:{batch_num}-Experiment:{exp_num})"
                    file_name = f"differences_plot_batch{batch_num}_experiment{exp_num}"

            elif len(y_true_cp.shape) == 2: 
                actuals = y_true[cycle_id, :]
                preds = y_pred[cycle_id, :]
                if batch_num == "one_batch":
                    title = f"SOH Degradation Plot - (Dataset:{dataset_name}-Experiment:{exp_num}-Cycle:{cycle_id})"
                    file_name = f"differences_plot_experiment{exp_num}_cycle{cycle_id}"
                else:
                    title = f"SOH Degradation Plot - (Dataset:{dataset_name}-Batch:{batch_num}-Experiment:{exp_num}-Cycle:{cycle_id})"
                    file_name = f"differences_plot_batch{batch_num}_experiment{exp_num}_cycle{cycle_id}"

            plt.plot(actuals, label='Actual SOH', alpha=.5, color='blue')
            plt.plot(preds, label='Predicted SOH', alpha=.5, color='red')
            plt.title(title)
            plt.xlabel('Cycle Index')
            plt.ylabel('SOH Value')
            plt.legend()
            plt.grid()

            if len(y_true_cp.shape) == 1:
                output_path = os.path.join(y_true_path.split('true_label.npy')[0], file_name)
            elif len(y_true_cp.shape) == 2: 
                output_path_root = os.path.join(y_true_path.split('true_label.npy')[0], 'plots')
                output_path = os.path.join(output_path_root, file_name)

            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            
            if len(y_true_cp.shape) == 1:
                print(f'Saved sucessfully (iteration:{i+1}): {output_path}')
            elif len(y_true_cp.shape) == 2: 
                print(f'Saved sucessfully (iteration:{i+1} - cycle:{cycle_id}): {output_path}')
