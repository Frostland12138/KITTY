import torch
import torchvision
from torch.utils import data
from custom_transfomers import augmentation
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
var_alpha=1.0
def read_in_csv(filename):
    sql_data=pd.read_csv(filename)
    sql_data=sql_data.values.tolist()
    return sql_data
def function2(_mean,_sigma,x):
    u1=_mean
    sigma1=_sigma
    return np.multiply(np.power(np.sqrt(2 * np.pi) * sigma1, -1), np.exp(-np.power(x - u1, 2) / 2 * sigma1 ** 2))

class MyDataSet(Dataset):
    def __init__(self, data_set_input,sampling):
        self.data_input = data_set_input
        self.length = len(data_set_input)
        print(self.length)
        all_nodes=[x for x in range(self.length)]
        self.new_samples=all_nodes
        if sampling=='KITTY':
            table=np.array(read_in_csv('./save/stable/stable-cifar-10/CIFAR_10_pretrained.csv'))
            samples=table[:,1]
            indexs=table[:,2]
            sigma=np.array([x*x for x in samples]).mean()-np.array(samples).mean()*np.array(samples).mean()
            sigma=np.sqrt(sigma)
            mean=samples.mean()

            alpha=var_alpha
            self.new_samples=[]
            for index,item in enumerate(samples):
                prob=1-alpha*function2(mean,sigma,item)
                if random.random()<prob:
                    self.new_samples.append(indexs[index])
            print(len(self.new_samples))

        if sampling=='Random':
            all_nodes=[x for x in range(self.length)]
            self.new_samples=random.choices(all_nodes,k=37761)
        
        self.length=len(self.new_samples)
        

    def __getitem__(self, mask):
        new_mask=int(self.new_samples[mask])
        data,label=self.data_input[new_mask]
        return data,label,new_mask

    def __len__(self):
        return self.length


def load_data(dataset, dataset_dir,  divide=False, trans1='weak', trans2='weak',sampling='',alpha=1.0):
    var_alpha=alpha
    if dataset == "CIFAR-10":
        p_strong = augmentation.create_config('./custom_transfomers/aug_strong_cifar10.yaml')
        strongtrans = augmentation.get_train_transformations(p_strong)

        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_cifar10.yaml')
        weakstrans = augmentation.get_train_transformations(p_weak)

        oral = augmentation.get_val_transformations(p_weak)

        def get_trans(name):
            if name == 'weak':
                return weakstrans
            if name == 'strong':
                return strongtrans
            if name == 'oral':
                return oral
            raise ValueError

        transform1 = get_trans(trans1)
        transform2 = get_trans(trans2)
        transform3 = get_trans(trans2)
        transform4 = get_trans(trans1)
        trans = augmentation.CreatePairTransforms(transform1, transform2)
        trans4 = augmentation.CreateQuarTransforms(oral, transform2,transform3,transform4)
        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            download=True,
            train=True,
            transform=trans4,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            download=True,
            train=False,
            transform=trans4,
        )
        train_dataset_val = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            download=True,
            train=True,
            transform=trans4,
        )
        test_dataset_val = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            download=True,
            train=False,
            transform=trans4,
        )
        if not divide:
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            datasetval = data.ConcatDataset([train_dataset_val, test_dataset_val])
        else:
            dataset = train_dataset
            datasetval = test_dataset_val
        dataset=MyDataSet(dataset,sampling)
        datasetval=MyDataSet(datasetval,sampling)
        class_num = 10
    elif dataset == "CIFAR-100":
        p_strong = augmentation.create_config('./custom_transfomers/aug_strong_cifar100.yaml')
        strongtrans = augmentation.get_train_transformations(p_strong)

        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_cifar100.yaml')
        weakstrans = augmentation.get_train_transformations(p_weak)

        oral = augmentation.get_val_transformations(p_weak)

        def get_trans(name):
            if name == 'weak':
                return weakstrans
            if name == 'strong':
                return strongtrans
            if name == 'oral':
                return oral
            raise ValueError

        transform1 = get_trans(trans1)
        transform2 = get_trans(trans2)
        transform3 = get_trans(trans2)
        transform4 = get_trans(trans1)
        trans = augmentation.CreatePairTransforms(transform1, transform2)
        trans4 = augmentation.CreateQuarTransforms(oral, transform2,transform3,transform4)
        train_dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            download=True,
            train=True,
            transform=trans4,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            download=True,
            train=False,
            transform=trans4,
        )
        train_dataset_val = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            download=True,
            train=True,
            transform=trans4,
        )
        test_dataset_val = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            download=True,
            train=False,
            transform=trans4,
        )
        if not divide:
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            datasetval = data.ConcatDataset([train_dataset_val, test_dataset_val])
        else:
            dataset = train_dataset
            datasetval = test_dataset_val
        dataset=MyDataSet(dataset,sampling)
        datasetval=MyDataSet(datasetval,sampling)
        class_num = 100
    elif dataset == "STL-10":
        p_strong = augmentation.create_config('./custom_transfomers/aug_strong_stl10.yaml')
        strongtrans = augmentation.get_train_transformations(p_strong)

        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_stl10.yaml')
        weakstrans = augmentation.get_train_transformations(p_weak)

        oral = augmentation.get_val_transformations(p_weak)

        def get_trans(name):
            if name == 'weak':
                return weakstrans
            if name == 'strong':
                return strongtrans
            if name == 'oral':
                return oral
            raise ValueError

        transform1 = get_trans(trans1)
        transform2 = get_trans(trans2)
        transform3 = get_trans(trans2)
        transform4 = get_trans(trans1)
        trans = augmentation.CreatePairTransforms(transform1, transform2)
        trans4 = augmentation.CreateQuarTransforms(oral, transform2,transform3,transform4)
        
        train_dataset = torchvision.datasets.STL10(
            root=dataset_dir,
            download=True,
            split='train',
            transform=trans4,
        )
        test_dataset = torchvision.datasets.STL10(
            root=dataset_dir,
            download=True,
            split='test',
            transform=trans4,
        )
        train_dataset_val = torchvision.datasets.STL10(
            root=dataset_dir,
            download=True,
            split='train',
            transform=trans4,
        )
        test_dataset_val = torchvision.datasets.STL10(
            root=dataset_dir,
            download=True,
            split='test',
            transform=trans4,
        )
        if not divide:
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            datasetval = data.ConcatDataset([train_dataset_val, test_dataset_val])
        else:
            dataset = train_dataset
            datasetval = test_dataset_val
        dataset=MyDataSet(dataset,sampling)
        datasetval=MyDataSet(datasetval,sampling)
        class_num = 10
    elif dataset == "tiny-ImageNet":
        p_strong = augmentation.create_config('./custom_transfomers/aug_strong_tiny_imagenet.yaml')
        strongtrans = augmentation.get_train_transformations(p_strong)

        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_tiny_imagenet.yaml')
        weakstrans = augmentation.get_train_transformations(p_weak)

        oral = augmentation.get_val_transformations(p_weak)

        def get_trans(name):
            if name == 'weak':
                return weakstrans
            if name == 'strong':
                return strongtrans
            if name == 'oral':
                return oral
            raise ValueError

        transform1 = get_trans(trans1)
        transform2 = get_trans(trans2)
        transform3 = get_trans(trans2)
        transform4 = get_trans(trans1)
        trans = augmentation.CreatePairTransforms(transform1, transform2)
        trans4 = augmentation.CreateQuarTransforms(oral, transform2,transform3,transform4)

        train_dataset = torchvision.datasets.ImageFolder(
            root='./dataset/tiny-imagenet-200/train',
            transform=trans4,
        )
        test_dataset_val = torchvision.datasets.ImageFolder(
            root='./dataset/tiny-imagenet-200/val/images',
            transform=trans4,
        )
        
        dataset = train_dataset
        datasetval = test_dataset_val
        dataset=MyDataSet(dataset,sampling)
        datasetval=MyDataSet(datasetval,sampling)
        class_num=200
    else:
        raise NotImplementedError

    return dataset, datasetval, class_num


def load_le_data(dataset, dataset_dir, workers=8, batchsize=256):
    if dataset == "CIFAR-10":
        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_cifar10.yaml')
        oral = augmentation.get_val_transformations(p_weak)

        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            download=True,
            transform=oral
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            download=True,
            transform=oral
        )
        class_num = 10
    elif dataset == "CIFAR-100":
        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_cifar100.yaml')
        oral = augmentation.get_val_transformations(p_weak)

        train_dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            train=True,
            download=True,
            transform=oral
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir,
            train=False,
            download=True,
            transform=oral
        )
        class_num = 100
    elif dataset == "STL-10":
        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_stl10.yaml')
        oral = augmentation.get_val_transformations(p_weak)

        train_dataset = torchvision.datasets.STL10(
            root=dataset_dir,
            split='train',
            download=True,
            transform=oral
        )
        test_dataset = torchvision.datasets.STL10(
            root=dataset_dir,
            split='test',
            download=True,
            transform=oral
        )
        class_num = 10

    elif dataset == "tiny-ImageNet":
        p_weak = augmentation.create_config('./custom_transfomers/aug_weak_tiny_imagenet.yaml')
        oral = augmentation.get_val_transformations(p_weak)

        train_dataset = torchvision.datasets.ImageFolder(
            root='./dataset/tiny-imagenet-200/train',
            transform=oral,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root='./dataset/tiny-imagenet-200/val',
            transform=oral,
        )
        class_num = 200
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
    )
    return train_loader, test_loader, class_num