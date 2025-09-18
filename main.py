import os
import sys
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision
import argparse
import pandas as pd
from tqdm import tqdm
from modules import resnet, contrastive_loss
from modules.network import Network
from utils import save_model, evaluation, load_data, load_model, \
     get_summarywriter, write_tensorboard_log, write_config,evaluation_,curvature
sys.path.append('./')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import random
from scipy import linalg as sclingan
def read_in_csv(filename):
    sql_data=pd.read_csv(filename)
    sql_data=sql_data.values.tolist()
    return sql_data
def file_name(file_dir):
    L = []
    path = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            xx = os.path.splitext(file)
            L.append(xx[0] + xx[1])
            path.append(root)
    return L, path
def write_csv(sql_data,filename):#
    #file_name = 'test1.csv'
    file_name=filename
    save = pd.DataFrame(sql_data)
    save.to_csv(file_name,encoding='utf_8_sig')
    return 0
def get_args():
    parser = argparse.ArgumentParser(description='Parameter Setting for train')
    # 
    parser.add_argument('--tables', default="", type=str, help='')
    parser.add_argument('--outputs', default="", type=str, help='')
    parser.add_argument('--mode', default="train_test", type=str, help='')
    parser.add_argument('--config', default="", type=str, help='Config File')

    parser.add_argument('--dataset', type=str, default='CIFAR-10',
                        help='Dataset Name, including CIFAR-10/CIFAR-100/STL-10/ImageNet-10/tiny-ImageNet/ImageNet-1k')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='path for dataset')
    parser.add_argument('--workers', type=int, default=8, help='worker num for dataloader')
    parser.add_argument('--seed', type=int, default=0, help='selected random seed')

    parser.add_argument('--trans1', type=str, default='weak',
                        help='first data augmentation level, including oral/weak/strong')
    parser.add_argument('--trans2', type=str, default='weak',
                        help='second data augmentation level, including oral/weak/strong')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size for training')
    parser.add_argument('--divide', action='store_true', help='whether to divide the dataset')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch for training')
    parser.add_argument('--epochs', type=int, default=802, help='epochs for training')
    parser.add_argument('--warm', type=int, default=0, help='epochs for warm up')
    parser.add_argument('--resnet', type=str, default='ResNet34',
                        help='backbone of the model, including ResNet18/ResNet34')
    parser.add_argument('--feature_dim', type=int, default=256, help='dim for ICH')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay for optimizer')

    parser.add_argument('--instance_temperature', type=float, default=0.2, help='temperature for instance loss')

    parser.add_argument('--model_path', type=str, default='./save', help='path to load and save model')
    parser.add_argument('--reload',default=False, action='store_true', help='whether to load pretrain model or reload checkpoint')
    parser.add_argument('--reload_model', type=str, default='checkpoint.tar', help='name of the load moedl')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')

    parser.add_argument('--experiment', type=str, default='simclr-CIFAR-10-with-random1.0-sampling', help='name of experiment')
    parser.add_argument('--name', type=str, default='dim256', help='name for tensorboard and save')
    parser.add_argument('--info', type=str, default='25', help='info for tensorboard and save')

    parser.add_argument('--ddp',default=False,action='store_true',help='whether to use DDP for multiprocess training')
    parser.add_argument('--sampling',default='Full_size',type=str,help='sampling for training sets')
    parser.add_argument('--alpha',default='1.0',type=float,help='alpha for kitty')
    parser.add_argument('--local_rank',type=int)
    args = parser.parse_args()

    return args


def set_selected_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
def function2(_mean,_sigma,x):
    u1=_mean
    sigma1=_sigma
    return np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))

if __name__ == "__main__":
    args = get_args()
    if args.seed == 0:
        args.seed = torch.initial_seed() % 2 ** 32
    write_config(args)
    set_selected_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.ddp:
        dist.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda',local_rank)
    else:
        device=torch.device('cuda:0')

    # set_selected_seed()
    dataset, datasetval, class_num = load_data(args.dataset, args.dataset_dir,  args.divide,
                                                         args.trans1, args.trans2,args.sampling,args.alpha)
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler = train_sampler
    )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
        )

    test_data_loader = torch.utils.data.DataLoader(
        datasetval,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = Network(res, args.feature_dim, class_num).to(device)
    if args.ddp:
        # 
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(model,device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print('Parallel Model Initialized')

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        load_model(args, model)
        
    criterion_instance_with_curv = contrastive_loss.InstanceLoss_with_curv(args.batch_size, args.instance_temperature, device).to(device)

    if (args.ddp and args.local_rank==0) or not args.ddp:
        writer = get_summarywriter('./tensorboard/{}/{}/{}_{}'.format(args.experiment, args.name, args.name, args.info))
    # train
    best = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.ddp:
            # 
            data_loader.sampler.set_epoch(epoch)
        
        model.train()
        loss_epoch = 0
        loss_ins = 0
        loss_clu = 0

        if (args.ddp and local_rank==0) or not args.ddp :
            data_enumerator = enumerate(tqdm(data_loader))
        else:
            data_enumerator = enumerate(data_loader)
        latents=[]
        couvatures=[]
        curvatures=[]
        labs=[]
        instance_indexes=[]
        for step, ((x_i, x_j,x_k,x_l), labt_,instance_index) in data_enumerator:
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            x_k = x_k.to(device)
            x_l = x_l.to(device)

            h_i, h_j = model(x_i, x_j, 3)
            h_k, h_l = model(x_k, x_l, 3)

            latents.append(h_i)
            labs.append(labt_)
            instance_indexes.append(instance_index)


            loss_instance,ccurv = criterion_instance_with_curv(h_i, h_j,h_k,h_l)
            curvatures.append(ccurv)

            loss = loss_instance

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if loss_instance != 0:
                loss_ins += loss_instance.item()
        if epoch%200==0:
            curs=[x.cpu().detach().numpy() for x in curvatures]
            curvatures=np.hstack(curs)
            indexes=[x.cpu().detach().numpy() for x in instance_indexes]
            indexes_all=np.hstack(indexes)
            sorted_curvatures=np.sort(curvatures)
            write_csv(np.vstack((curvatures,indexes_all)).T,'./save/'+args.experiment+str(epoch)+'.csv')
            for index,item in enumerate(sorted_curvatures):
                writer.add_scalar('curvature_distribution_epoch'+str(epoch),item,index)
        
        loss_info = [
            [loss_epoch / len(data_loader), 'loss_all'],
            [loss_ins / len(data_loader), 'loss_i'],
            [loss_clu / len(data_loader), 'loss_c']
        ]
        print(loss_info)
            
        write_tensorboard_log(writer, loss_info, loss_info, epoch)
        print(epoch)

        if args.ddp and args.local_rank==0:
            if (epoch + 1) % 50 == 0:
                save_model(args, model.module, optimizer, epoch, args.experiment, args.name, args.info + "_{}".format(epoch+1))
        else:
            if (epoch + 1) % 50 == 0:
                save_model(args, model, optimizer, epoch, args.experiment, args.name, args.info + "_{}".format(epoch+1))
    writer.flush()
    writer.close()

