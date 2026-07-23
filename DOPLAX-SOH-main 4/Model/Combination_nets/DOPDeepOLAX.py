import torch                 
import torch.nn as nn        
import numpy as np         
from utils.util import get_logger
from Model.utils.lr_schedulers import LR_Scheduler
from Model.utils.util import AverageMeter, eval_metrix 
from Model.Auxiliary_nets.Solution_u import Solution_u
from Model.Auxiliary_nets.MLP import MLP
from Model.PI_nets import DeepOPINN 
from Model.Combination_nets import DeepOLAX
from Model.PI_nets.LAX import OptimizationNetwork 
import os 
import warnings 
warnings.filterwarnings('ignore') 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
import deepxde as dde



class MLP_Bagging_NN(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, hidden_dim=[50], dropout=0.2):
        super(MLP_Bagging_NN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = len(hidden_dim) + 1
        self.hidden_dim = hidden_dim
        assert self.layers_num >= 2, "layers must be greater than 2"

        self.layers = []
        for i in range(self.layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim[i]))
                self.layers.append(nn.Tanh())
            elif i == self.layers_num - 1:
                self.layers.append(nn.Linear(hidden_dim[i-1], output_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
                self.layers.append(nn.Tanh())
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)


    def forward(self,x):
        x = self.net(x)
        return x


        
class Model(DeepOLAX.Model):
    def __init__(self, args):
        super(Model, self).__init__(args=None)
        self.args = args
        # Create CUDA generator for DataLoaders
        self.generator = torch.Generator(device=device)
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        self.log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(self.log_dir)
        
        if self.args.data == "XJTU":
            # related to DeepOPINN
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
            # related to LAX network
            self.args.betha_LAX = args.betha_LAX_XJTU
            self.args.dual_LAX = args.dual_LAX_XJTU
            self.args.theta_LAX = args.theta_LAX_XJTU
            self.args.lr_net = args.lr_net_LAX_XJTU
            self.args.h_dim_LAX = args.h_dim_LAX_XJTU
            self.args.beta_LAX = args.beta_LAX_XJTU
            self.args.center_block_LAX = args.center_block_LAX_XJTU
            self.args.distance_block_LAX = args.distance_block_LAX_XJTU
            self.args.time_block_LAX = args.time_block_LAX_XJTU
            # related to combination network
            self.args.bagging_NN_lr = self.args.bagging_NN_lr_XJTU
            self.args.bag_hidden_dim = self.args.bag_hidden_dim_XJTU
            self.args.mono_bag = self.args.mono_bag_XJTU
        elif self.args.data == "TJU":
            # related to DeepOPINN
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
            # related to LAX network
            self.args.betha_LAX = args.betha_LAX_TJU
            self.args.dual_LAX = args.dual_LAX_TJU
            self.args.theta_LAX = args.theta_LAX_TJU
            self.args.lr_net = args.lr_net_LAX_TJU
            self.args.h_dim_LAX = args.h_dim_LAX_TJU
            self.args.beta_LAX = args.beta_LAX_TJU
            self.args.center_block_LAX = args.center_block_LAX_TJU
            self.args.distance_block_LAX = args.distance_block_LAX_TJU
            self.args.time_block_LAX = args.time_block_LAX_TJU
            # related to combination network
            self.args.bagging_NN_lr = self.args.bagging_NN_lr_TJU
            self.args.bag_hidden_dim = self.args.bag_hidden_dim_TJU
            self.args.mono_bag = self.args.mono_bag_TJU
        elif self.args.data == "MIT":
            # related to DeepOPINN
            self.args.lr = self.args.lr_MIT
            self.args.warmup_lr = self.args.warmup_lr_MIT
            self.args.lr_F = self.args.lr_F_MIT
            self.args.final_lr = self.args.final_lr_MIT
            self.args.F_hidden_dim = self.args.F_hidden_dim_MIT
            self.args.F_layers_num = self.args.F_layers_num_MIT
            self.args.dropout = self.args.dropout_MIT
            self.args.alpha = self.args.alpha_MIT
            self.args.beta = self.args.beta_MIT
            self.args.beta_LAX = args.beta_LAX_MIT
            self.args.warmup_epochs = self.args.warmup_epochs_MIT
            # related to LAX network
            self.args.betha_LAX = args.betha_LAX_MIT
            self.args.dual_LAX = args.dual_LAX_MIT
            self.args.theta_LAX = args.theta_LAX_MIT
            self.args.lr_net = args.lr_net_LAX_MIT
            self.args.h_dim_LAX = args.h_dim_LAX_MIT
            self.args.beta_LAX = args.beta_LAX_MIT
            self.args.center_block_LAX = args.center_block_LAX_MIT
            self.args.distance_block_LAX = args.distance_block_LAX_MIT
            self.args.time_block_LAX = args.time_block_LAX_MIT
            # related to combination network
            self.args.bagging_NN_lr = self.args.bagging_NN_lr_MIT
            self.args.bag_hidden_dim = self.args.bag_hidden_dim_MIT
            self.args.mono_bag = self.args.mono_bag_MIT
        else:
            # related to DeepOPINN
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
            # related to LAX network
            self.args.betha_LAX = args.betha_LAX_HUST
            self.args.dual_LAX = args.dual_LAX_HUST
            self.args.theta_LAX = args.theta_LAX_HUST
            self.args.lr_net = args.lr_net_LAX_HUST
            self.args.h_dim_LAX = args.h_dim_LAX_HUST
            self.args.beta_LAX = args.beta_LAX_HUST
            self.args.center_block_LAX = args.center_block_LAX_HUST
            self.args.distance_block_LAX = args.distance_block_LAX_HUST
            self.args.time_block_LAX = args.time_block_LAX_HUST
            # related to combination network
            self.args.bagging_NN_lr = self.args.bagging_NN_lr_HUST
            self.args.bag_hidden_dim = self.args.bag_hidden_dim_HUST
            self.args.mono_bag = self.args.mono_bag_HUST
            
        self._save_args()

        # Act as F
        self.solution_u = Solution_u(
            layers_num=self.args.F_layers_num,
            hidden_dim=60,
            dropout=self.args.dropout
        ).to(device)
        # Act as G
        if self.args.F_hidden_dim <= 50:
            hidden_dim_F = self.args.F_hidden_dim
        else:
            hidden_dim_F = self.args.F_hidden_dim*2
        self.dynamical_F_deepopinn = self.dynamical_F_deepolax = MLP(input_dim=35,output_dim=1, 
                               layers_num=self.args.F_layers_num, 
                               hidden_dim=hidden_dim_F, 
                               dropout=self.args.dropout).to(device)

        self.bagging_NN = MLP_Bagging_NN(input_dim=2, output_dim=1,
                                         hidden_dim=self.args.bag_hidden_dim,  
                                         dropout=self.args.dropout).to(device)

        self.extractor_deepopinn = None
        self.extractor_deepolax = None
        self.LAX_model = None
        self.deepopinn = DeepOPINN.Model(args=self.args, save_args=False)
        
        self.optimizer_extractor = None
        self.optimizer_solution = None
        self.bagging_NN_optimizer = None
        self.scheduler = None
        
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
        
        
    def extract_features(self, x, extractor): 
        B, L = x.shape
        metadata = x[:, -1].unsqueeze(1)
        coordinates = torch.linspace(0, 1, self.m-1).reshape(-1, 1).to(device) 
        inputs = (x, coordinates)
        features = extractor(inputs)
        return torch.cat([features, metadata], dim=1)

        
    def Test(self, testloader, epoch=None):
        self.eval()
        true_label = []
        pred_label = []
        current = self.args.run_for_LAX
        setattr(self.args, 'run_for_LAX', False)

        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(device)
                xt1 = self.extract_features(x1, extractor=self.extractor_deepopinn)
                _, u_pinn = self.deepopinn.predict(xt1, solution_u=self.solution_u)
                xt1 = self.extract_features(x1, extractor=self.extractor_deepolax)
                u_lax = self.LAX_model(x=xt1[:,:-1], t=xt1[:,-1], epoch=epoch)
                u1 = self.bagging_NN(torch.cat([u_pinn, u_lax], dim=1))
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        setattr(self.args, 'run_for_LAX', current)

        return true_label,pred_label


    def Valid(self, validloader, epoch=None):
        self.eval()
        true_label = []
        pred_label = []
        current = self.args.run_for_LAX
        setattr(self.args, 'run_for_LAX', False)
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(device)
                xt1 = self.extract_features(x1, extractor=self.extractor_deepopinn)
                _, u_pinn = self.deepopinn.predict(xt1, solution_u=self.solution_u)
                xt1 = self.extract_features(x1, extractor=self.extractor_deepolax)
                u_lax = self.LAX_model(x=xt1[:,:-1], t=xt1[:,-1], epoch=epoch)
                u1 = self.bagging_NN(torch.cat([u_pinn, u_lax], dim=1))
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        setattr(self.args, 'run_for_LAX', current)

        return mse.item()

    
    def load_DeepOPINN_model(self, model_path, freeze):
        # Load checkpoint to CPU first
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize extractor if it doesn't exist
        if self.extractor_deepopinn is None:
            try:
                self.m = checkpoint['input_dimention']
            except:
                self.m = find_input_dim_from_checkpoint(checkpoint_path=model_path)
            layer_sizes_branch = [self.m] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
            layer_sizes_trunk = [self.args.dim_x] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
            
            self.extractor_deepopinn = dde.nn.DeepONetCartesianProd(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation="relu",
                kernel_initializer="Glorot normal"
            ).to(device)
        self.extractor_deepopinn.load_state_dict(checkpoint['feature_extractor'], strict=False)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F_deepopinn.load_state_dict(checkpoint['dynamical_F'])
        
        for param in self.extractor_deepopinn.parameters():
            param.requires_grad = not freeze
            
        for param in self.solution_u.parameters():
            param.requires_grad = not freeze


    def forward_bagging_NN(self, U1, U2):
        bnn = self.bagging_NN(torch.cat([U1, U2], dim=1))
        return bnn


    def forward_DeepOLAX(self, xt, epoch):
        xt = self.extract_features(xt, extractor=self.extractor_deepolax)
        bnn = self.LAX_model(x=xt[:,:-1], t=xt[:,-1], epoch=epoch)
        return bnn


    def train_one_epoch(self, epoch, dataloader):
        self.train()

        loss_data_bagging_meter = AverageMeter()
        loss_bagging_monotone_meter = AverageMeter()


        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            
            if self.extractor_deepopinn is None and self.extractor_deepolax is None:
                self.m = x1.shape[1]
                layer_sizes_branch = [self.m] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
                layer_sizes_trunk = [self.args.dim_x] + [self.args.F_hidden_dim] * (self.args.F_layers_num - 1)
                self.extractor_deepopinn = self.extractor_deepolax = dde.nn.DeepONetCartesianProd(
                    layer_sizes_branch=layer_sizes_branch, 
            	    layer_sizes_trunk=layer_sizes_trunk,
            	    activation="relu",
            	    kernel_initializer="Glorot normal"
            	).to(device)
                
            # Forward pass
            u1, f1 = self.deepopinn.forward_deepopinn(x1, m=self.m, extractor=self.extractor_deepopinn, solution_u=self.solution_u, dynamical_F=self.dynamical_F_deepopinn) # forward_deepopinn
            u2, f2 = self.deepopinn.forward_deepopinn(x2, m=self.m, extractor=self.extractor_deepopinn, solution_u=self.solution_u, dynamical_F=self.dynamical_F_deepopinn) # forward_deepopinn

            u1_lax = self.forward_DeepOLAX(x1, epoch=epoch)
            u2_lax = self.forward_DeepOLAX(x2, epoch=epoch)

            Bag_u1 = self.forward_bagging_NN(u1, u1_lax)
            Bag_u2 = self.forward_bagging_NN(u2, u2_lax)

            # loss_data_Bagging
            loss_data_bagging = 0.5*self.loss_func(Bag_u1, y1) + 0.5*self.loss_func(Bag_u2, y2)

            # loss_Bagging_monotone
            loss_bagging_monotone = self.relu(torch.mul(Bag_u2-Bag_u1, y1-y2)).sum()

                
            # total loss
            loss = loss_data_bagging + self.args.mono_bag*loss_bagging_monotone

            # Backward pass with separate optimizers
            self.bagging_NN_optimizer.zero_grad()

            loss.backward()
            
            # Step each optimizer separately
            self.bagging_NN_optimizer.step()

            loss_data_bagging_meter.update(loss_data_bagging.item())
            loss_bagging_monotone_meter.update(loss_bagging_monotone.item())

            # Logging
            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, physics loss:{:.6f}".format(epoch, iter+1, loss_data_bagging, loss_bagging_monotone))

        return loss_data_bagging_meter.avg, loss_bagging_monotone_meter.avg


    def Train(self, trainloader, validloader=None, testloader=None):
        # Recreate dataloaders with CUDA-safe sampler
        if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'generator'):
            trainloader.sampler.generator = self.generator

        if hasattr(validloader, 'sampler') and hasattr(validloader.sampler, 'generator'):
            validloader.sampler.generator = self.generator
            
        if type(self.args.run_for_DeepOPINN) == bool:
            self.load_DeepOPINN_model(model_path=self.args.DOP_weights, freeze=not self.args.run_for_DeepOPINN)

        self.LAX_model = self.get_LAX_model_instance(trainloader)
        if type(self.args.run_for_LAX) == bool:
            self.load_DeepOLAX_model(model_path=self.args.LAX_weights, freeze=not self.args.run_for_LAX, trainloader=trainloader)

        self.bagging_NN_optimizer = torch.optim.Adam([ # for bagging
            {"params": self.extractor_deepopinn.parameters(), "lr": self.args.warmup_lr*0.1},
            {"params": self.extractor_deepolax.parameters(), "lr": self.args.warmup_lr*0.1},
            {"params": self.solution_u.parameters(), "lr": self.args.warmup_lr},
            {"params": self.LAX_model.parameters(), "lr": self.args.lr_net},
            {"params": self.bagging_NN.parameters(), "lr": self.args.bagging_NN_lr}
        ])
        self.scheduler = LR_Scheduler( # for bagging
            optimizer=self.bagging_NN_optimizer, 
            warmup_epochs=self.args.warmup_epochs,
            warmup_lr=self.args.warmup_lr,
            num_epochs=self.args.epochs,
            base_lr=self.args.lr,
            final_lr=self.args.final_lr
        )

        min_valid_mse = np.inf
        valid_mse = 10
        early_stop = 0

        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss_data_bagging, loss_bagging_monotone = self.train_one_epoch(e, trainloader) # forward_deepopinn
            current_lr = self.scheduler.step()
            total_loss = loss_data_bagging  + self.args.mono_bag*loss_bagging_monotone
            self.train_loss_history.append(total_loss)
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, current_lr, total_loss)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader, epoch=e)

                self.valid_loss_history.append(valid_mse)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)

            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader, epoch=e)
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
                    'extractor_deepopinn': self.extractor_deepopinn.state_dict(),
                    'extractor_deepolax': self.extractor_deepolax.state_dict(),
                    'solution_u': self.solution_u.state_dict(),
                    'LAX': self.LAX_model.state_dict(),
                    'bagging_u': self.bagging_NN.state_dict() # currently added
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
