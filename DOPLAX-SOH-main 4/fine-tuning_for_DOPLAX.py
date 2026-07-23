import os
import numpy as np
import torch
import torch.nn as nn
from dataloader.dataloader import XJTUdata, TJUdata
from dataloader.data_helper import load_MIT_data, load_HUST_data
from Model.Combination_nets import Bagging_u
from Model.utils.lr_schedulers import LR_Scheduler
from Model.utils.util import AverageMeter, eval_metrix
from utils.util import write_to_file
from utils.arguments import get_finetuning_Bagging_u_args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = torch.Generator(device=device)

# mapping dictionaries for model filenames
XJTU_BATCH_NAMES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
TJU_BATCH_NAMES  = ['NCA', 'NCM', 'NCM_NCA']



def load_XJTU_data(args, small_sample=None):
    root = os.path.join('data', 'Full', 'XJTU data')
    batch_names= ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    batch_num = args.target_batch if args.target_data == 'XJTU' else args.source_batch
    batch = batch_names[batch_num]
    args.batch = batch
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader


def load_TJU_data(args, small_sample=None):
    root = os.path.join('data', 'Full', 'TJU data')
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    mod = [(5,9),(4,8),(5,9)]
    batchs = os.listdir(root)
    batch_num = args.target_batch if args.target_data == 'TJU' else args.source_batch
    batch = batchs[batch_num]
    args.batch = batch
    batch_root = os.path.join(root,batch)
    files = os.listdir(batch_root)
    for i,f in enumerate(files):
        id = i + 1
        if id % 10 == mod[batch_num][0] or id % 10 == mod[batch_num][1]:
            test_list.append(os.path.join(batch_root,f))

        else:
            train_list.append(os.path.join(batch_root,f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader
    

class AdaModel(Bagging_u.Model):
    def __init__(self, args, trainloader):
        super(AdaModel, self).__init__(args=args)
        self.args = args
        self.load_DOPLAX_model(model_path=self.args.pretrained_model, freeze=False, trainloader=trainloader)
        
        self.ada_optimizer = torch.optim.Adam([
            {"params": self.extractor_deepopinn.parameters(), "lr": self.args.warmup_lr*0.1},
            {"params": self.solution_u.parameters(), "lr": self.args.warmup_lr},
            {"params": self.LAX_model.parameters(), "lr": self.args.lr_net},
            {"params": self.bagging_NN.parameters(), "lr": self.args.bagging_NN_lr}
        ])
        self.scheduler = LR_Scheduler(
            optimizer=self.ada_optimizer, 
            warmup_epochs=self.args.warmup_epochs,
            warmup_lr=self.args.warmup_lr,
            num_epochs=self.args.epochs,
            base_lr=self.args.lr,
            final_lr=self.args.final_lr
        )

    def adaptation_one_epoch(self, epoch, dataloader):
        self.bagging_NN.train()

        loss_data_bagging_meter = AverageMeter()
        loss_bagging_monotone_meter = AverageMeter()
        
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            
            u1, f1 = self.forward_deepopinn(x1, m=self.m, extractor=self.extractor_deepopinn, solution_u=self.solution_u, dynamical_F=self.dynamical_F)
            u2, f2 = self.forward_deepopinn(x2, m=self.m, extractor=self.extractor_deepopinn, solution_u=self.solution_u, dynamical_F=self.dynamical_F)

            u1_lax = self.forward_LAX(x1, epoch=epoch)
            u2_lax = self.forward_LAX(x2, epoch=epoch)

            Bag_u1 = self.forward_bagging_NN(u1, u1_lax)
            Bag_u2 = self.forward_bagging_NN(u2, u2_lax)

            # loss_data_Bagging
            loss_data_bagging = 0.5*self.loss_func(Bag_u1, y1) + 0.5*self.loss_func(Bag_u2, y2)

            # loss_Bagging_monotone
            loss_bagging_monotone = self.relu(torch.mul(Bag_u2-Bag_u1, y1-y2)).sum()

            # total loss
            loss = loss_data_bagging + self.args.mono_bag*loss_bagging_monotone
        
            # self.extractor_optimizer.zero_grad()
            self.ada_optimizer.zero_grad()
            loss.backward()
            # self.extractor_optimizer.step()
            self.ada_optimizer.step()

            loss_data_bagging_meter.update(loss_data_bagging.item())
            loss_bagging_monotone_meter.update(loss_bagging_monotone.item())
            
            debug_info = "[epoch:{} iter:{}] data loss:{:.6f}, physics loss:{:.6f}".format(epoch, iter+1, loss_data_bagging, loss_bagging_monotone)
    
            if epoch < 3:
                self.logger.debug(debug_info)

            # Logging
            if (iter+1) % 50 == 0:
                print(debug_info)

        return loss_data_bagging_meter.avg, loss_bagging_monotone_meter.avg


    def Adaptation(self,trainloader,validloader=None,testloader=None):
        # if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'generator'):
        #     trainloader.sampler.generator = self.generator

        # if hasattr(validloader, 'sampler') and hasattr(validloader.sampler, 'generator'):
        #     validloader.sampler.generator = self.generator

        for param in self.extractor_deepopinn.parameters():
            param.requires_grad = False

        for param in self.solution_u.parameters():
            param.requires_grad = False
            
        for param in self.LAX_model.parameters():
            param.requires_grad = False
            
        for param in self.dynamical_F.parameters(): # freeze the dynamical_F
            param.requires_grad = False

        # min_valid_mse = 10
        min_valid_mse = np.inf
        valid_mse = 10
        early_stop = 0

        for e in range(1, self.args.epochs+1):
            early_stop += 1
            loss_data_bagging, loss_bagging_monotone = self.adaptation_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            total_loss = loss_data_bagging  + self.args.mono_bag*loss_bagging_monotone
            self.train_loss_history.append(total_loss)
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, current_lr, total_loss)
            self.logger.info(info)

            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader, epoch=e)
                self.valid_loss_history.append(valid_mse)
                info = '[Valid] epoch:{}, MSE: {}'.format(e,valid_mse)
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
                    'feature_extractor': self.extractor_deepopinn.state_dict(),
                    'solution_u': self.solution_u.state_dict(),
                    'dynamical_F': self.dynamical_F.state_dict(),
                    'LAX': self.LAX_model.state_dict(),
                    'bagging_u': self.bagging_NN.state_dict() # currently added
                }
                
                if self.args.save_folder is not None:
                    y_true_path = os.path.join(self.args.save_folder, 'true_label.npy')
                    y_pred_path = os.path.join(self.args.save_folder, 'pred_label.npy')
                    np.save(y_true_path, true_label)
                    np.save(y_pred_path, pred_label)

            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
                
        self._save_loss_history_plot()
        self._save_model()
        self._clear_logger()


def one_adaptation_task(args, pretrain_model_path, source, target, source_batch=-1, target_batch=-1):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    if source == 'XJTU':
        suffix = XJTU_BATCH_NAMES[source_batch]
        model_dir = os.path.join(pretrain_model_path, f'DOPLAX_model_XJTU_{suffix}.pth')
    elif source == 'TJU':
        suffix = TJU_BATCH_NAMES[source_batch]
        model_dir = os.path.join(pretrain_model_path, f'DOPLAX_model_TJU_{suffix}.pth')
    else:  # MIT or HUST
        model_dir = os.path.join(pretrain_model_path, f'DOPLAX_model_{source}.pth')
    
    setattr(args,'pretrained_model',model_dir)
    setattr(args,'source_data',source)
    setattr(args,'data',source)
    setattr(args,'source_batch',source_batch)
    setattr(args,'target_data',target)
    setattr(args,'target_batch',target_batch)

    if args.target_data == 'XJTU':
        args.batch_size = args.batch_size_XJTU
    elif args.target_data == 'TJU':
        args.batch_size = args.batch_size_TJU
    elif args.target_data == 'MIT':
        args.batch_size = args.batch_size_MIT
    else:
        args.batch_size = args.batch_size_HUST
        
    # load data
    args.run_mode = "DeepOPINN" 
    target_loader = eval(f'load_{target}_data')(args, small_sample=1)

    if hasattr(target_loader['train'], 'sampler') and hasattr(target_loader['train'].sampler, 'generator'):
        target_loader['train'].sampler.generator = generator

    if hasattr(target_loader['valid'], 'sampler') and hasattr(target_loader['valid'].sampler, 'generator'):
        target_loader['valid'].sampler.generator = generator
            
    # load model
    model = AdaModel(args, trainloader=target_loader['train'])

    # Firstly, test source model in target domain
    true_label,pred_label = model.Test(target_loader['test'], epoch=1)
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)

    print('Before adaptation (source only):')
    print('MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE))
    if args.log_dir is not None and args.save_folder is not None:
        save_name = os.path.join(args.save_folder,args.log_dir)
        info = 'Source only: {} -> {} | MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(source,target,MSE, MAE, MAPE, RMSE)
        write_to_file(save_name, info)

    # adaptation
    model.Adaptation(trainloader=target_loader['train'], validloader=target_loader['valid'], testloader=target_loader['test'])


def FineTune_TJU2XJTU(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_Bagging_u_args()
    source_batchs = [2] # [0, 1, 2]
    target_batchs = [0] # [0, 0, 0]
    for sb, tb in zip(source_batchs, target_batchs):
        for experiment in range(10):
            setattr(args, 'log_dir', 'logging.txt')
            save_folder = os.path.join(save_folder_path, 'TJU-XJTU', f'batch{tb}', f'Experiment{experiment}')
            setattr(args, 'save_folder', save_folder)
            one_adaptation_task(args, pretrain_model_path, source='TJU', target='XJTU', source_batch=sb, target_batch=tb)


def FineTune_XJTU2TJU(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_Bagging_u_args()
    source_batchs = [0] # [0, 4]
    target_batchs = [2] # [2, 2] 
    for sb, tb in zip(source_batchs, target_batchs):
        for experiment in range(10):
            # setattr(args, 'warmup_lr', lr)
            save_folder = os.path.join(save_folder_path, 'XJTU-TJU', f'batch{tb}', f'Experiment{experiment}')
            setattr(args, 'save_folder', save_folder)
            one_adaptation_task(args, pretrain_model_path, source='XJTU', target='TJU', source_batch=sb, target_batch=tb)


def FineTune_HUST2MIT(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_Bagging_u_args()
    for experiment in range(10):
        setattr(args, 'log_dir', 'logging.txt')
        save_folder = os.path.join(save_folder_path, 'HUST-MIT', f'Experiment{experiment}')
        setattr(args, 'save_folder', save_folder)
        one_adaptation_task(args, pretrain_model_path, source='HUST', target='MIT')


def FineTune_MIT2HUST(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_Bagging_u_args()
    for experiment in range(10):
        setattr(args, 'log_dir', 'logging.txt')
        save_folder = os.path.join(save_folder_path, 'MIT-HUST', f'Experiment{experiment}')
        setattr(args, 'save_folder', save_folder)
        one_adaptation_task(args, pretrain_model_path, source='MIT', target='HUST')


def FineTune(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_Bagging_u_args()
    datasets = ['XJTU', 'TJU', 'HUST', 'MIT']
    batchs = [0, 2, -1, -1]
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            if source in ['XJTU', 'TJU'] and target in ['XJTU', 'TJU']:
                continue
            if source in ['HUST', 'MIT'] and target in ['HUST', 'MIT']:
                continue
            if source == 'XJTU' or source == 'TJU' or source == 'HUST':
                continue
                
            sb = batchs[i]
            tb = batchs[j]

            for e in range(10):
                setattr(args, 'log_dir', f'logging.txt')
                save_folder = os.path.join(save_folder_path, f'{source}-{target}', f'Experiment{e + 1}')
                setattr(args, 'save_folder', save_folder)
                one_adaptation_task(args, pretrain_model_path, source=source, target=target, source_batch=sb, target_batch=tb)


if __name__ == '__main__':
    pretrain_model_path = os.path.join('pretrained_models', 'DOPLAX-MLP')
    save_folder_path = os.path.join('results_fine-tuning', 'DOPLAX-MLP')
    
    FineTune(pretrain_model_path, save_folder_path)
    FineTune_MIT2HUST(pretrain_model_path, save_folder_path)
    FineTune_HUST2MIT(pretrain_model_path, save_folder_path)
    FineTune_TJU2XJTU(pretrain_model_path, save_folder_path)
    FineTune_XJTU2TJU(pretrain_model_path, save_folder_path)
