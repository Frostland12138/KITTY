import os
import argparse
import torch
import torchvision
import numpy as np
from utils import evaluation_, load_le_data, write_config, load_model,\
    get_summarywriter, write_tensorboard_log
from modules import resnet, contrastive_loss
from modules.network import Network


def get_args():
    parser = argparse.ArgumentParser(description='Parameter Setting for linear evaluation')
    parser.add_argument('--dataset', type=str, default='CIFAR-100',
                        help='Dataset Name, including CIFAR-10/CIFAR-100/STL-10/ImageNet-10/tiny-ImageNet')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='path for dataset')
    parser.add_argument('--workers', type=int, default=8, help='worker num for dataloader')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs for training')
    parser.add_argument('--resnet', type=str, default='ResNet34',
                        help='backbone of the model, including ResNet18/ResNet34')
    parser.add_argument('--feature_dim', type=int, default=256, help='dim')

    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay for optimizer')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch for training')

    parser.add_argument('--logdir', type=str, default='./new_tensorboard/linear_evaluation',
                        help='path for save tensorboard information')
    parser.add_argument('--model_path', type=str, default='', help='path of the pretrain moedl')
    parser.add_argument('--reload_model', type=str, default='checkpoint_dim256_25_600.tar', help='name of the reload moedl')
    parser.add_argument('--save_model_path', type=str,
                        default='',
                        help='path of the save moedl')

    parser.add_argument('--experiment', type=str, default='CIFAR-10-random-sampling', help='name of experiment')
    parser.add_argument('--name', type=str, default='-linear_evaluation-', help='name for tensorboard and save')
    parser.add_argument('--info', type=str, default='256', help='info for tensorboard and save')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--ebd', type=str, default='ebd', help='target of linear evaluation, including ebd/pjt')
    args = parser.parse_args()

    return args


def linearfc_train(loader, model, criterion, optimizer, device):
    model.train()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        if step% 100 ==0:
            print(step)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            embedding = model.forward_embedding(x)
        c = model.forward_linearevaluton_fc(embedding)
        loss = criterion(c, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        feature_vector.extend(torch.argmax(c, dim=1).cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())


    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def linearfc_test(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        # print(len(x),x[0].shape)
        x = x.to(device)
        with torch.no_grad():
            embedding = model.forward_embedding(x)
            c = model.forward_linearevaluton_fc(embedding)
        feature_vector.extend(torch.argmax(c, dim=1).cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    return feature_vector, labels_vector


if __name__ == "__main__":
    args = get_args()
    write_config(args)
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:1")

    train_loader, test_loader, class_num = load_le_data(args.dataset, args.dataset_dir, args.workers, args.batch_size)

    res = resnet.get_resnet(args.resnet)
    model = Network(res, args.feature_dim, class_num)
    load_model(args, model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    linearfc = torch.nn.Linear(model.resnet.rep_dim, class_num).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    bestacc = 0

    writer = get_summarywriter('./new_tensorboard/{}/{}/{}_{}'.format(args.experiment, args.name, args.name, args.info))

    for epoch in range(args.epochs):
        predict_train, Y_train = linearfc_train(train_loader, model, criterion, optimizer, device)
        predict_test, Y_test = linearfc_test(test_loader, model, device)
        nmi_train, ari_train, acc_train = evaluation_(Y_train, predict_train)

        print('\ntrain epoch: {}'.format(epoch))
        print('NMI = {:.4f} ARI = {:.4f} ACC = {:.4f}'.format(nmi_train, ari_train, acc_train))
        print('test::{}'.format(epoch))
        nmi_test, ari_test, acc_test = evaluation_(Y_test, predict_test)
        if acc_test > bestacc:
            bestacc = acc_test
        print('NMI = {:.4f} ARI = {:.4f} ACC = {:.4f}'.format(nmi_test, ari_test, acc_test))
        print('bestacc:', bestacc)

        eval_info = [
            [acc_train, 'train acc'],
            [ari_train, 'train ari'],
            [nmi_train, 'train nmi'],
            [acc_test, 'test acc'],
            [ari_test, 'test ari'],
            [nmi_test, 'test nmi'],
        ]
        loss_info = []
        write_tensorboard_log(writer, eval_info, loss_info, epoch)
    torch.save(model, args.save_model_path)
