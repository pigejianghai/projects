import torch
from torchvision import models
from torch.utils.data import DataLoader
from Net import Net
from MyDataLoader import MyDataset
import time
import os
import argparse
import warnings
from sklearn.model_selection import train_test_split
from focal_loss import focal_loss
import torchmetrics as tm
from torch.utils.tensorboard import SummaryWriter
from utils import LRScheduler, EarlyStoppingAUC

def id_label_list(txt_path):
    id_list, label_list = [], []
    f = open(txt_path, 'r')
    for line in f:
        id_list.append(line.split()[0])
        label_list.append(line.split()[1])
    return id_list, label_list

def training(device, net, optimizer, criterion, train_loader):
    training_loss = 0.0
    training_corrects = 0
    net.train()
    strat = time.time()
    total_correct = 0

    train_acc = tm.Accuracy(task='binary').to(device)

    for x, y in train_loader:   # x是110*110*3图片，y是标签
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        
        probability, y_pre = torch.max(y_hat, 1)
        total_correct += sum(y_pre == y.data)    
        
        loss = criterion(y_hat, y)  

        loss.backward()
        optimizer.step()

        training_loss += loss.item() * x.size(0)
        training_corrects += torch.sum(y_pre == y)

        train_acc(y_pre, y)
    epoch_loss = training_loss / len(train_data)
    total_acc = train_acc.compute()
    print(f'Training loss: {epoch_loss:.4f} Training Acc: {total_acc:.4f}')

    writer.add_scalar('ACC/Train_Acc', total_acc, global_step=epoch)
    writer.add_scalar('LOSS/Train_Loss', loss.item(), global_step=epoch)
    train_acc.reset()
    
def validating(device, net, criterion, val_loader, best_auc, checkpoint_name):
    net.eval()
    validating_loss = 0.0
    validating_corrects = 0
    best_auc = best_auc

    val_acc = tm.Accuracy(task='binary').to(device)
    val_recall = tm.Recall(task='binary', average='none', num_classes=2).to(device)
    val_precision = tm.Precision(task='binary', average='none', num_classes=2).to(device)
    val_auc = tm.AUROC(task='binary', average='none', num_classes=2).to(device)

    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)

            output = net(image)
            probability, preds = torch.max(output, 1)
            loss = criterion(output, label)

            validating_loss += loss.item() * image.size(0)
            validating_corrects += torch.sum(preds == label.data)       
            
            val_acc(preds, label)
            val_auc.update(probability, label)
            val_recall(preds, label)
            val_precision(preds, label)
        epoch_loss = validating_loss / len(test_data)
        
        total_acc = val_acc.compute()
        total_recall = val_recall.compute()
        total_precision = val_precision.compute()
        total_auc = val_auc.compute()
        epoch_auc = total_auc.item()

        if epoch_auc > best_auc and epoch > 10:
            best_auc = epoch_auc
            torch.save(net.state_dict(), checkpoint_name)

        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        #     f"Avg loss: {test_loss:>8f}, "
        #     f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        print(f'Validating loss: {epoch_loss:.4f}')
        print("\taccuracy of every validating dataset class: ", total_acc)
        print("\trecall of every validating dataset class: ", total_recall)
        print("\tprecision of every validating dataset class: ", total_precision)
        print("\tvalidating auc:", total_auc.item())

        writer.add_scalar('AUC/Val_Auc', total_auc.item(), global_step=epoch)
        writer.add_scalar('LOSS/Val_Loss', epoch_loss, global_step=epoch)
        writer.add_scalar('ACC/Val_Acc', total_acc, global_step=epoch)

    # 清空计算对象
        val_precision.reset()
        val_acc.reset()
        val_recall.reset()
        val_auc.reset()

        return best_auc, epoch_auc

def testing(device, net, criterion, test_loader):
    net.eval()
    testing_loss = 0.0
    testing_corrects = 0

    test_acc = tm.Accuracy(task='binary').to(device)
    test_recall = tm.Recall(task='binary', average='none', num_classes=2).to(device)
    test_precision = tm.Precision(task='binary', average='none', num_classes=2).to(device)
    test_auc = tm.AUROC(task='binary', average='none', num_classes=2).to(device)

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = net(image)
            probability, preds = torch.max(output, 1)
            loss = criterion(output, label)

            testing_loss += loss.item() * image.size(0)
            testing_corrects += torch.sum(preds == label.data)       
            
            test_acc(preds, label)
            test_auc.update(probability, label)
            test_recall(preds, label)
            test_precision(preds, label)

        epoch_loss = testing_loss / len(test_data)

        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        total_auc = test_auc.compute()

        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        #     f"Avg loss: {test_loss:>8f}, "
        #     f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        print(f'Testing loss: {epoch_loss:.4f}')
        print("\taccuracy of every testing dataset class: ", total_acc)
        print("\trecall of every testing dataset class: ", total_recall)
        print("\tprecision of every testing dataset class: ", total_precision)
        print("\ttesting auc:", total_auc.item())

        writer.add_scalar('AUC/Test_Auc', total_auc.item(), global_step=epoch)
        writer.add_scalar('LOSS/Test_Loss', epoch_loss, global_step=epoch)
        writer.add_scalar('ACC/Test_Acc', total_acc, global_step=epoch)

    # 清空计算对象
        test_precision.reset()
        test_acc.reset()
        test_recall.reset()
        test_auc.reset()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    criterion = focal_loss(gamma=3).to(device)#alpha=[0.556, 0.444], 
    date_name = '11211705_net'
    checkpoint_fold = '/data1/jianghai/DECT/checkpoint/' + date_name
    if not os.path.exists(checkpoint_fold):
        os.mkdir(checkpoint_fold)
    root = '/data1/jianghai/DECT/txt/binary/'
    batch_size = 8
    epochs = 300
    lr_init = 1e-4
    weight_decay = 1e-2
    writer = SummaryWriter()
    test_size = 0.25
    random_state = 42

    test_id, test_label = id_label_list(root + 'test.txt')
    test_data = MyDataset(id_list=test_id, label_list=test_label, flag='test')
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1, shuffle=False)

    id_list, label_list = id_label_list(root + 'train.txt')
    train_id, val_id, train_label, val_label = train_test_split(id_list, label_list, 
                                                                stratify=label_list, test_size=test_size, 
                                                                random_state=random_state)
    train_data = MyDataset(id_list=train_id, label_list=train_label, flag='train')
    train_loader = DataLoader(dataset=train_data, num_workers=1, pin_memory=True, batch_size=batch_size, shuffle=True)
    val_data = MyDataset(id_list=val_id, label_list=val_label, flag='test')
    val_loader = DataLoader(dataset=val_data, num_workers=1, pin_memory=True, batch_size=1, shuffle=False)
    
    net = Net()
    # net = models.resnet50(num_classes=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_init, weight_decay=weight_decay)
    net.to(device)

    patience = 2
    min_lr = lr_init * 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
    args = vars(parser.parse_args())
    if args['lr_scheduler']:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer, patience=patience, min_lr=min_lr)
    if args['early_stopping']:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStoppingAUC(patience=patience)

    print('criterion: ', criterion, ' batch_size: ', batch_size)
    print('epochs: ', epochs, ' learning_rage: ', lr_init)
    print('early_stop patience: ', patience, ' minimum_lr: ', min_lr)
    print(net)
    
    val_epoch_auc = 0.0
    val_auc = []
    best_auc = 0.0
    for epoch in range(epochs):
        print('Epoch : ', epoch)
        training(device=device, net=net, optimizer=optimizer, criterion=criterion, train_loader=train_loader)

        checkpoint_name = checkpoint_fold + os.sep + str(epoch) + '.pth.tar'
        best_auc, val_epoch_auc = validating(device=device, net=net, criterion=criterion, val_loader=test_loader, best_auc=best_auc, 
                                             checkpoint_name=checkpoint_name)

        val_auc.append(val_epoch_auc)
        if args['lr_scheduler']:
            lr_scheduler(val_epoch_auc)
        if args['early_stopping']:
            early_stopping(val_epoch_auc)
            if early_stopping.early_stop:
                break

        testing(device=device, net=net, criterion=criterion, test_loader=test_loader)
        print('---------------------------------------------------------------')
        writer.flush()