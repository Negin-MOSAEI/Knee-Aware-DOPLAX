import os
import numpy as np
import torch
import torch.nn as nn
from dataloader.dataloader import XJTUdata, TJUdata
from dataloader.data_helper import load_MIT_data, load_HUST_data
from Model import PINN
from Model.utils.lr_schedulers import LR_Scheduler
from Model.utils.util import AverageMeter, eval_metrix
from utils.util import write_to_file
from utils.arguments import get_finetuning_PINN_args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mapping dictionaries for model filenames
XJTU_BATCH_NAMES = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
TJU_BATCH_NAMES  = ['NCA', 'NCM', 'NCM_NCA']



def load_XJTU_data(args, small_sample=None):
    root = os.path.join('data', 'XJTU data')
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
    root = os.path.join('data', 'TJU data')
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


class AdaModel(PINN.Model):
    def __init__(self,args):
        super(AdaModel, self).__init__(args)
        self.load_model(model_path=args.pretrained_model, freeze=False)
        self.ada_optimizer = torch.optim.Adam(
            self.solution_u.parameters(),
            lr=args.adaptation_lr
        )

    def adaptation_one_epoch(self, epoch, dataloader):
        self.solution_u.train()

        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()

        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1, y1) + 0.5*self.loss_func(u2, y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1, f_target) + 0.5*self.loss_func(f2, f_target)

            # physics loss  u2-u1<0, considering capacity regeneration effect
            loss3 = self.relu(torch.mul(u2-u1, y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
                         "PDE loss:{:.6f}, physics loss:{:.6f}, " \
                         "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            if epoch < 3:
                self.logger.debug(debug_info)

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))

        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg


    def Adaptation(self,trainloader,validloader=None,testloader=None):
        # if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'generator'):
        #     trainloader.sampler.generator = self.generator

        # if hasattr(validloader, 'sampler') and hasattr(validloader.sampler, 'generator'):
        #     validloader.sampler.generator = self.generator
            
        for param in self.dynamical_F.parameters(): # freeze the dynamical_F
            param.requires_grad = False

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss1, loss2, loss3 = self.adaptation_one_epoch(e, trainloader)
            info = '[Train] epoch:{}, data loss:{:.6f}, ' \
                   'PDE loss:{:.6f}, ' \
                   'physics loss:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, loss1, loss2, loss3,
                                              loss1 + self.alpha * loss2 + self.beta * loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
                
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))
        self._clear_logger()


def one_adaptation_task(args, pretrain_model_path, source, target, source_batch=-1, target_batch=-1):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if source == 'XJTU':
        suffix = XJTU_BATCH_NAMES[source_batch]
        model_dir = os.path.join(pretrain_model_path, f'PINN_model_XJTU_{suffix}.pth')
    elif source == 'TJU':
        suffix = TJU_BATCH_NAMES[source_batch]
        model_dir = os.path.join(pretrain_model_path, f'PINN_model_TJU_{suffix}.pth')
    else:  # MIT or HUST
        model_dir = os.path.join(pretrain_model_path, f'PINN_model_{source}.pth')

    setattr(args,'pretrained_model',model_dir)
    setattr(args,'source_data',source)
    setattr(args,'data',source)
    setattr(args,'target_data',target)
    setattr(args,'target_batch',target_batch)

    # if args.target_data == 'XJTU':
    #     args.batch_size = args.batch_size_XJTU
    # elif args.target_data == 'TJU':
    #     args.batch_size = args.batch_size_TJU
    # elif args.target_data == 'MIT':
    #     args.batch_size = args.batch_size_MIT
    # else:
    #     args.batch_size = args.batch_size_HUST
        
    # load data
    args.run_mode = "PINN" 
    target_loader = eval(f'load_{target}_data')(args, small_sample=1)

    # load model
    model = AdaModel(args)

    # Firstly, test source model in target domain
    true_label,pred_label = model.Test(target_loader['test'])
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
    args = get_finetuning_PINN_args()
    lrs = [0.0004, 0.01, 0.0005, 0.002, 0.003, 0.0006]
    source_batchs = [2, 2, 2, 1, 0, 1]
    target_batchs = [0, 1, 2, 3, 4, 5]
    for lr,sb,tb in zip(lrs,source_batchs,target_batchs):
        for experiment in range(10):
            setattr(args, 'adaptation_lr', lr)
            setattr(args, 'log_dir', 'logging.txt')
            save_folder = os.path.join(save_folder_path, 'TJU-XJTU', f'batch{tb}', f'Experiment{experiment+1}')
            setattr(args, 'save_folder', save_folder)
            one_adaptation_task(args, pretrain_model_path, source='TJU', target='XJTU', source_batch=sb, target_batch=tb)


def FineTune_XJTU2TJU(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_PINN_args()
    lrs = [0.003, 0.002, 0.002]
    source_batchs = [3, 3, 2]
    target_batchs = [0, 1, 2]
    for lr, sb, tb in zip(lrs, source_batchs, target_batchs):
        for experiment in range(10):
            setattr(args, 'adaptation_lr', lr)
            setattr(args, 'log_dir', 'logging.txt')
            save_folder = os.path.join(save_folder_path, 'XJTU-TJU', f'batch{tb}', f'Experiment{experiment+1}')
            setattr(args, 'save_folder', save_folder)
            one_adaptation_task(args, pretrain_model_path, source='XJTU', target='TJU', source_batch=sb, target_batch=tb)


def FineTune_HUST2MIT(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_PINN_args()
    for experiment in range(10):
        setattr(args, 'adaptation_lr', 0.005)
        setattr(args, 'log_dir', 'logging.txt')
        save_folder = os.path.join(save_folder_path, 'HUST-MIT', f'Experiment{experiment+1}')
        setattr(args, 'save_folder', save_folder)
        one_adaptation_task(args, pretrain_model_path, source='HUST', target='MIT')


def FineTune_MIT2HUST(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_PINN_args()
    for experiment in range(10):
        setattr(args, 'adaptation_lr', 0.0002)
        setattr(args, 'log_dir', 'logging.txt')
        save_folder = os.path.join(save_folder_path, 'MIT-HUST', f'Experiment{experiment+1}')
        setattr(args, 'save_folder', save_folder)
        one_adaptation_task(args, pretrain_model_path, source='MIT', target='HUST')


def FineTune(pretrain_model_path:str, save_folder_path:str):
    args = get_finetuning_PINN_args()
    datasets = ['XJTU', 'TJU', 'HUST', 'MIT']
    batchs = [0, 2, -1, -1]
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            if source in ['XJTU', 'TJU'] and target in ['XJTU', 'TJU']:
                continue
            if source in ['HUST', 'MIT'] and target in ['HUST', 'MIT']:
                continue
            sb = batchs[i]
            tb = batchs[j]

            for e in range(10):
                setattr(args, 'adaptation_lr', 0.001)
                setattr(args, 'log_dir', f'logging.txt')
                save_folder = os.path.join(save_folder_path, f'{source}-{target}', f'Experiment{e + 1}')
                setattr(args, 'save_folder', save_folder)
                one_adaptation_task(args, pretrain_model_path, source=source, target=target, source_batch=sb, target_batch=tb)



if __name__ == '__main__':
    pretrain_model_path = os.path.join('pretrained_models', 'PINN')
    save_folder_path = os.path.join('results_fine-tuning', 'PINN-XJTU2TJU')

    FineTune(pretrain_model_path, save_folder_path)
    FineTune_MIT2HUST(pretrain_model_path, save_folder_path)
    FineTune_HUST2MIT(pretrain_model_path, save_folder_path)
    FineTune_TJU2XJTU(pretrain_model_path, save_folder_path)
    FineTune_XJTU2TJU(pretrain_model_path, save_folder_path)
