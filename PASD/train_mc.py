import torch.optim as optim
import torch
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import warnings
# import argparse
import torchvision.models as models
import random
# import torchmetrics as tm
import torchmetrics.classification as tmc
import timm
# from linformer import Linformer
from dataset_mt import Mri_Loader, get_id_label
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
# from vit_pytorch.efficient import ViT
from datetime import datetime
# from CNNs import InstanceNet, InstanceNet2d
from torch.utils.tensorboard import SummaryWriter
def seed_torch(seed=1101):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enable = True

def training(train_loader, len_train, net, criterion, epoch, writer, 
             device, 
             fold, optimizer, num_classes):
    training_loss = 0.0
    net.train()

    train_acc = tmc.MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    
    train_auc = tmc.MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    # train_auc_mc = tmc.MulticlassAUROC(num_classes=num_classes, average='none').to(device)
    
    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)
        outputs_hat = F.softmax(outputs, dim=1)
        
        # _, preds = torch.max(outputs_hat, dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item() * images.size(0)
        
        train_acc.update(outputs_hat, labels)
        
        train_auc.update(outputs_hat, labels)
        # train_auc_mc.update(outputs_hat, labels)

    epoch_loss = training_loss / len_train
    
    total_acc = train_acc.compute()
    
    total_auc = train_auc.compute()
    # total_auc_mc = train_auc_mc.compute()
    
    print('Training:')
    print('\t'f'AUC  : {total_auc.item():.4f}')
    # print('\t'f'AUC of each: {total_auc_mc.item():.4f}')
    print('\t'f'Loss : {epoch_loss:.4f}\tAcc: {total_acc:.4f}')

    writer.add_scalar('Acc/train_acc_'+str(fold), total_acc, global_step=epoch)
    writer.add_scalar('Loss/train_loss_'+str(fold), loss.item(), global_step=epoch)
    writer.add_scalar('AUC/train_AUC_'+str(fold), total_auc.item(), global_step=epoch)
    # writer.add_scalar('AUC/train_AUC_each_'+str(fold), total_auc_mc.item(), global_step=epoch)
    
    train_acc.reset()
    train_auc.reset()
    # train_auc_mc.reset()

def validating(val_loader, len_val, net, criterion, epoch, writer, 
               device, 
               best_auc, best_acc, fold, num_classes, checkpoint_name):#
    net.eval()
    val_loss = 0.0
    best_auc = best_auc
    best_acc = best_acc

    val_acc = tmc.MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    # val_recall = tmc.MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    # val_precision = tmc.MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    # val_f1score = tmc.MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    val_auc = tmc.MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    # val_auc_mc = tmc.MulticlassAUROC(num_classes=num_classes, average='none').to(device)
    
    val_cm = tmc.MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = net(images)
            loss = criterion(output, labels)
            output_hat = F.softmax(output, dim=1)
            
            # _, preds = torch.max(output, dim=1)
            
            val_loss += loss.item() * images.size(0)
            val_acc.update(output_hat, labels)
            # val_recall.update(output_hat, labels)
            # val_precision.update(output_hat, labels)
            # val_f1score.update(output_hat, labels)
            val_auc.update(output_hat, labels)
            # val_auc_mc.update(output_hat, labels)
            val_cm.update(output_hat, labels)
            
        epoch_loss = val_loss / len_val

        total_acc = val_acc.compute()
        # total_recall = val_precision.compute()
        # total_precision = val_precision.compute()
        # total_f1s = val_f1score.compute()
        total_auc = val_auc.compute()
        # total_auc_mc = val_auc_mc.compute()
        total_cm = val_cm.compute()
        
        print('Validating')
        print('\t'f'AUC   : {total_auc.item():.4f}')
        # print('\t'f'AUC of each: {total_auc_mc.item():.4f}')
        print('\t'f'Loss  : {epoch_loss:.4f}\tAcc: {total_acc:.4f}')
        # print('\t'f'Recall: {total_recall:.4f}\tPrecision: {total_precision:.4f}\tF1_score: {total_f1s:.4f}')
        print(total_cm)

        if total_auc.item() > best_auc and epoch > 10:
            best_auc = total_auc.item()
            torch.save(net.state_dict(), checkpoint_name)
        # if total_acc > best_acc and epoch > 10:
        #     best_acc = total_acc
            # torch.save(net.state_dict(), checkpoint_name)
        writer.add_scalar('AUC/val_AUC_'+str(fold), total_auc.item(), global_step=epoch)
        # writer.add_scalar('AUC/val_AUC_each_'+str(fold), total_auc_mc.item(), global_step=epoch)
        writer.add_scalar('Acc/val_acc_'+str(fold), total_acc, global_step=epoch)
        writer.add_scalar('Loss/val_loss_'+str(fold), loss.item(), global_step=epoch)
        # val_precision.reset()
        val_acc.reset()
        # val_recall.reset()
        val_auc.reset() 
        # val_auc_mc.reset()
    
    return epoch_loss, best_auc, total_auc.item(), best_acc

def testing(test_loader, net, criterion, epoch, writer, device, fold):
    net.eval()
    testing_loss = 0.0
    testing_corrects = 0

    test_acc = tmc.Accuracy(task='multiclass').to(device)
    test_recall = tmc.Recall(task='multiclass').to(device)
    test_precision = tmc.Precision(task='multiclass').to(device)
    test_auc = tmc.AUROC(task='multiclass', average='macro').to(device)
    with torch.no_grad():
        for image, label in test_loader:#, _, mask
            image = image.to(device)
            label = label.to(device)
            # mask = mask.to(device)

            output = net(image)
            probs, preds = torch.max(output, 1)
            loss = criterion(output, label)
            # loss = criterion(output, label, predicted_roi, mask)

            testing_loss += loss.item() * image.size(0)
            testing_corrects += torch.sum(preds == label.data)

            test_acc.update(probs, label)
            test_auc.update(probs, label)
            test_recall.update(probs, label)
            test_precision.update(probs, label)
        # fpr, tpr, thresholds = metrics.roc_curve(v_true, v_pred)
        # accuracy = metrics.accuracy_score(test_true, test_pred)
        # epoch_loss = testing_loss / len(test_data) 
        # epoch_acc = testing_corrects.double() / len(test_data)

        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        total_auc = test_auc.compute()
        # print(f'Testing Loss: {epoch_loss:.4f}')
        print('\tacc of every testing dataset class: ', total_acc)
        print('\trecall of every testing dataset class: ', total_recall)
        print('\tprecision of every testing dataset class: ', total_precision)
        print('\tauc of every testing dataset class: ', total_auc.item())

        writer.add_scalar('AUC/test_auc_'+str(fold), total_auc.item(), global_step=epoch)
        writer.add_scalar('Acc/test_acc_'+str(fold), total_acc, global_step=epoch)
        writer.add_scalar('Loss/test_loss_'+str(fold), loss.item(), global_step=epoch)
        test_acc.reset()
        test_recall.reset()
        test_precision.reset()
        test_auc.reset()

if __name__ == '__main__':
    ###-----------Hyperparameters-----------###
    ###########################################
    RANDOM_SEED = 2024
    seed_torch(seed=RANDOM_SEED)
    DATE_NET = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_PATH = '/data1/jianghai/open_source/neoadjuvant'
    checkpoint_fold = ROOT_PATH + os.sep + 'checkpoint' + os.sep + DATE_NET + '_resnet34'
    if not os.path.exists(checkpoint_fold):
        os.mkdir(checkpoint_fold)
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    LR_INIT = 1e-4
    # lr_stepsize = 20
    GAMMA = 0.7
    WEIGHT_DECAY = 1e-2
    DEVICE = torch.device('cuda:0')
    print(DEVICE)
    CRITERION = torch.nn.CrossEntropyLoss()
    NUM_SPLITS = 5
    EPOCHS = 500
    writer = SummaryWriter()
    print('batch_size:', BATCH_SIZE, ' num_classes:', NUM_CLASSES, ' epochs:', EPOCHS)
    print('learning_rate_initial:', LR_INIT, ' weight_decay: ', WEIGHT_DECAY)#, ' lr_stepsize:', lr_stepsize, ' weight_decay:', weight_decay)
    print('criterion: ', CRITERION)
    warnings.filterwarnings("ignore")
    path_list, label_list = get_id_label("/data1/jianghai/open_source/neoadjuvant/ViTransformers/multi_task/txt/type_multi_class_4.txt")
    # path_list_train, path_list_test, label_list_train, label_list_test = train_test_split(path_list, label_list, stratify=label_list, 
    #                                                                                       test_size=0.33, random_state=RANDOM_SEED)

    skf = StratifiedKFold(n_splits=NUM_SPLITS, random_state=RANDOM_SEED, shuffle=True)
    for fold, (train_index, val_index) in enumerate(skf.split(path_list, label_list)):
        train_path_list, train_label_list = [], []
        val_path_list, val_label_list = [], []
        for t in train_index:
            train_path_list.append(path_list[t])
            train_label_list.append(label_list[t])
        for v in val_index:
            val_path_list.append(path_list[v])
            val_label_list.append(label_list[v])
            
        MODEL = models.resnet34(num_classes=NUM_CLASSES)#weights=models.ResNet18_Weights
        # MODEL = timm.create_model('maxvit_tiny_tf_tf_')
        # MODEL = torch.nn.DataParallel(MODEL, device_ids=[0, 1])
        # fc_inputs = MODEL.fc.in_features
        # MODEL.fc = torch.nn.Linear(fc_inputs, NUM_CLASSES)
        # MODEL = torch.nn.DataParallel(MODEL)
        MODEL = MODEL.to(DEVICE)
        # NET = NET.to(DEVICE)
        # EFFICIENT_TRANSFORMER = Linformer(
        #     dim=128, 
        #     seq_len=257, 
        #     depth=12, 
        #     heads=16, 
        #     k=256
        # )
        # MODEL = ViT(
        #     dim=128, 
        #     image_size=448, 
        #     patch_size=28, 
        #     num_classes=2, 
        #     transformer=EFFICIENT_TRANSFORMER, 
        #     channels=10, 
        # ).to(DEVICE)

        OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LR_INIT)
        SCHEDULER = optim.lr_scheduler.StepLR(optimizer=OPTIMIZER, step_size=1, gamma=GAMMA)
        # optimizer = optim.Adam(NET.parameters(), lr=LR_INIT)#, weight_decay=WEIGHT_DECAY
    ###########################################
    ################################
    ###-----------Data-----------###
    
        train_data = Mri_Loader(path_list=train_path_list, label_list=train_label_list, flag='train')   
        train_loader = DataLoader(dataset=train_data, 
                                  batch_size=BATCH_SIZE, 
                                  num_workers=1, 
                                  pin_memory=True, 
                                  shuffle=True)
        val_data = Mri_Loader(path_list=val_path_list, label_list=val_label_list, flag='val')
        val_loader = DataLoader(dataset=val_data, 
                                num_workers=1, 
                                pin_memory=True, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)
        # test_data = Mri_Loader(path_list=path_list_test, label_list=label_list_test, flag='test')
        # test_loader = DataLoader(dataset=test_data, 
        #                          num_workers=1, 
        #                          pin_memory=True, 
        #                          batch_size=BATCH_SIZE, 
        #                          shuffle=False)
    ###-----------Data-----------###
    ################################
    ###-----------Training && Test-----------###
    ############################################
        # val_auc = []
        best_auc = 0.0
        best_acc = 0.0
        val_epoch_auc = 0.0
        for epoch in range(EPOCHS):
            checkpoint_name = checkpoint_fold + os.sep + 'Fold_' + str(fold) + '_Epoch_' + str(epoch) + '.pth.tar'
            print('-----------------------------------------------------------------------')
            print('Epoch: ', epoch)
            training(train_loader=train_loader, len_train=len(train_data), 
                     net=MODEL, criterion=CRITERION, 
                     epoch=epoch, writer=writer, 
                     device=DEVICE, 
                     fold=fold, 
                     optimizer=OPTIMIZER, num_classes=NUM_CLASSES)
            #######################################
            #######################################
            val_epoch_loss, best_auc, val_epoch_auc, best_acc = validating(val_loader=val_loader, len_val=len(val_data), 
                                                                           net=MODEL, criterion=CRITERION, 
                                                                           epoch=epoch, writer=writer, 
                                                                           device=DEVICE, 
                                                                           checkpoint_name=checkpoint_name, 
                                                                           best_auc=best_auc, best_acc=best_acc, 
                                                                           fold=fold, num_classes=NUM_CLASSES)
            # val_auc.append(val_epoch_auc)
            # if args['lr_scheduler']:
            #     lr_scheduler(val_epoch_auc)
            # if args['early_stopping']:
            #     early_stopping(val_epoch_auc)
            #     if early_stopping.early_stop:
            #         break
            #######################################
            #######################################
            # testing(test_loader=test_loader, net=NET, criterion=CRITERION, 
            #         epoch=epoch, writer=writer, device=DEVICE, fold=fold)
            # print()
    writer.flush()
    ############################################
    ###-----------Training && Test-----------###