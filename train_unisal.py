import argparse
import random
from sklearn.manifold import TSNE
import torch
import sys
import pandas as pd
import numpy as np
import os
import time
import logging
import copy
from networks.resnet_my import ResNet50_fc, ResNet50_fc_mapping
from networks.vgg import VGG16_fc_mapping
from tqdm import tqdm
from dataset.alb_dataset import Tumor_dataset_two_weak, Tumor_dataset_val_cls, get_loader, get_train_loader_ssl
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import sklearn.metrics as metrics
from PIL import Image
import matplotlib.pyplot as plt


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_files(data_csv):
    data = pd.read_csv(data_csv)
    data_name = data.iloc[:, 0]
    data_label = data.iloc[:, 1]
    data_label = np.array(data_label).astype(np.uint8)
    data_name = data_name.to_list()
    new_file = [{"img": img, "label": label} for img, label in zip(data_name, data_label)]
    new_dict = {k:v for k, v in zip(data_name, data_label)}
    return new_file

def get_arguments():
    parser = argparse.ArgumentParser(
        description="xxxx Pytorch implementation")
    parser.add_argument("--num_class", type=int, default=9, help="Train class num")
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--batch_size", type=int, default=128, help="Train batch size")
    parser.add_argument("--labeled_bs", type=int, default=64)
    parser.add_argument("--num_workers", default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=8e-4)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--rounds", default=10, type=int)
    parser.add_argument("--query_num", type=int, default=180)
    parser.add_argument("--epochs", default=200, type=int)

    return parser.parse_args()

def infonce_loss(features1, features2, temperature=0.07):
    """
    Compute InfoNCE loss (contrastive loss) for SimCLR.
    
    Args:
        features1 (torch.Tensor): Features from the first view, shape [batch_size, feature_dim].
        features2 (torch.Tensor): Features from the second view, shape [batch_size, feature_dim].
        temperature (float): Temperature parameter for scaling the similarity.
        
    Returns:
        torch.Tensor: The InfoNCE loss value.
    """
    # Step 1: Normalize the feature vectors
    features1 = F.normalize(features1, p=2, dim=-1)  # Normalize features1
    features2 = F.normalize(features2, p=2, dim=-1)  # Normalize features2
    
    # Step 2: Compute the similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(features1, features2.T)  # Shape [batch_size, batch_size]
    
    # Step 3: Compute the logits by scaling with temperature
    logits = similarity_matrix / temperature  # Scale by temperature parameter
    
    # Step 4: Create labels for positive pairs: diagonal entries are the positive pairs
    labels = torch.arange(features1.size(0)).long().cuda()  # Positive pairs on the diagonal
    
    # Step 5: Compute the loss using cross entropy
    loss = F.cross_entropy(logits, labels)
    
    return loss

def train(train_loader, validation_loader, test_loader, l, round, args):
    # load model
    model1 = ResNet50_fc_mapping(pretrain=True, num_classes=args.num_class).cuda()
    # model1 = VGG16_fc_mapping(pretrain=True).cuda() 
    model1.train()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3, weight_decay=8e-4, momentum=0.9)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4, weight_decay=8e-4, amsgrad=True)

    model2 = ResNet50_fc_mapping(pretrain=True, num_classes=args.num_class).cuda()
    # model2 = VGG16_fc_mapping(pretrain=True).cuda() 
    model2.train()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3, weight_decay=8e-4, momentum=0.9)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4, weight_decay=8e-4, amsgrad=True)

    train_epochs_per_round = args.epochs
    max_val_accuracy = 0
    max_epoch = -1
    best_model = None
    train_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()

    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[train_epochs_per_round//2, train_epochs_per_round*3//4])
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[train_epochs_per_round//2, train_epochs_per_round*3//4])

    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[train_epochs_per_round*3//4])
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[train_epochs_per_round*3//4])

    for epoch in tqdm(range(train_epochs_per_round+1)):
        train_accuracy = 0
        model1.train()
        model2.train()
        threshold = 0.7
        with torch.cuda.amp.autocast():
            for counter, sample in enumerate(train_loader):
                x1 = sample['img1'].cuda()
                x2 = sample['img2'].cuda()
                y_batch = sample['label'].cuda().long()

                logits1, feature1 = model1(x1, True, True)
                logits2, feature2 = model2(x2, True, True)

                logits_soft1, logits_soft2 = torch.softmax(logits1, dim=1), torch.softmax(logits2, dim=1)

               # consistency
                if epoch < 15:
                    consistency_weight = 0
                else:
                    # 0.2 is nice, 0.5 is better than 0.2, 1 is worse than 0.5. 0.5 is a good value. 0.75 is also good.
                    consistency_weight = 0.5

                loss1 = criterion(logits1[:args.labeled_bs], y_batch[:args.labeled_bs])
                loss2 = criterion(logits2[:args.labeled_bs], y_batch[:args.labeled_bs])

                pseudo_idx1 = (torch.max(logits_soft1[args.labeled_bs:], dim=1)[0] > threshold).long()
                pseudo_idx2 = (torch.max(logits_soft2[args.labeled_bs:], dim=1)[0] > threshold).long()

                pseudo_outputs1 = torch.argmax(logits_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(logits_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = torch.mean(F.cross_entropy(logits1[args.labeled_bs:], pseudo_outputs2, reduction='none') * pseudo_idx2)
                pseudo_supervision2 = torch.mean(F.cross_entropy(logits2[args.labeled_bs:], pseudo_outputs1, reduction='none') * pseudo_idx1)

                # use all samples, use -1 to 
                pseudo_labels_confidence = torch.zeros_like(y_batch).cuda() - 1
                pseudo_labels_confidence[:args.labeled_bs] = y_batch[:args.labeled_bs]
                for i in range(pseudo_idx1.shape[0]):
                    if pseudo_idx1[i] == 1 and pseudo_idx2[i] == 1 and pseudo_outputs1[i] == pseudo_outputs2[i]:
                        pseudo_labels_confidence[args.labeled_bs+i] = pseudo_outputs1[i]
                # print(pseudo_labels_confidence)
                T = 0.15
                dis_avg = 0
                count_pos = 1
                # print(pseudo_labels_confidence)
                for i in range(feature2.shape[0]-1):
                    cur_feature_w = feature2[i].unsqueeze(0)
                    cur_feature_s = feature1[i].unsqueeze(0)
                    cur_neg = 0
                    cur_pos = F.cosine_similarity(cur_feature_w, cur_feature_s)
                    cur_pos = (cur_pos/T).exp()
                    # print(cur_pos)
                    if pseudo_labels_confidence[i] == -1:
                        continue
                    for j in range(i+1, feature2.shape[0]-1):
                        # if pseudo_labels_confidence[j] == pseudo_labels_confidence[i]:
                        #     dis_cur += 1 - F.cosine_similarity(cur_feature, feature2[j].unsqueeze(0))
                        if pseudo_labels_confidence[j] != pseudo_labels_confidence[i] and pseudo_labels_confidence[j] != -1:
                            cur_neg += (F.cosine_similarity(cur_feature_w, feature2[j].unsqueeze(0))/T).exp()
                            cur_neg += (F.cosine_similarity(cur_feature_s, feature1[j].unsqueeze(0))/T).exp()
                            # cur_neg += (F.cosine_similarity(cur_feature_w, feature1[j].unsqueeze(0))/T).exp()
                            # cur_neg += (F.cosine_similarity(cur_feature_s, feature2[j].unsqueeze(0))/T).exp()
                            # cur_neg += F.cosine_similarity(cur_feature_w, feature1[j].unsqueeze(0))
                            # cur_neg += F.cosine_similarity(cur_feature_s, feature2[j].unsqueeze(0))
                    if cur_neg != 0:
                        # print(cur_pos, cur_neg)
                        dis = -torch.log(cur_pos/(cur_pos+cur_neg))
                        dis_avg += dis
                        count_pos += 1
                dis = dis_avg/count_pos

                model1_loss = loss1 + consistency_weight * pseudo_supervision1
                model2_loss = loss2 + consistency_weight * pseudo_supervision2

                loss = model1_loss + model2_loss + 0.1*dis
                # loss = model1_loss + model2_loss
                
                top1 = accuracy(logits1, y_batch, topk=(1,))
                train_accuracy += top1[0]

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer1)
                scaler.step(optimizer2)
                scaler.update()

            train_accuracy /= (counter + 1)
            val_accuracy = 0
        # print(dis_loss.item()/(counter+1))
        model1.eval()
        model2.eval()
        with torch.no_grad():
            pred_all, gt_all = torch.zeros((1, )), torch.zeros((1, ))
            for counter, sample in enumerate(validation_loader):
                x_batch = sample['img'].cuda()
                y_batch = sample['label'].cuda().long()

                logits = (model1(x_batch)+model2(x_batch))/2
                # print(y_batch, logits)
                top1 = accuracy(logits, y_batch, topk=(1,))
                val_accuracy += top1[0]

                logits_hard = torch.argmax(logits, dim=1)
                gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
                pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)
        pred_all, gt_all = pred_all[1:], gt_all[1:]
        y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)

        val_accuracy = metrics.accuracy_score(y_true, y_pred)

        scheduler1.step()
        scheduler2.step()

        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_epoch = epoch
            best_model1 = copy.deepcopy(model1)
            best_model2 = copy.deepcopy(model2)

    l.info("Training has finished.")
    # save model checkpoints
    # l.info(f"The best performing epoch is {max_epoch}, max validation accuracy {max_val_accuracy}.")

    test_accuracy = 0
    best_model1.eval()
    best_model2.eval()
    with torch.no_grad():
        pred_all, gt_all = torch.zeros((1, )), torch.zeros((1, ))
        # pred, gt = np.zeros((7,)), np.zeros((7,))
        for counter, sample in enumerate(test_loader):
            x_batch = sample['img']
            y_batch = sample['label']
            x_batch = x_batch.type(torch.FloatTensor)
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda().long()

            logits = (best_model1(x_batch)+best_model2(x_batch))/2

            top1 = accuracy(logits, y_batch, topk=(1,))
            test_accuracy += top1[0]
            logits_hard = torch.argmax(logits, dim=1)

            gt_all = torch.cat((gt_all, y_batch.cpu()), dim=0)
            pred_all = torch.cat((pred_all, logits_hard.cpu()), dim=0)

        pred_all, gt_all = pred_all[1:], gt_all[1:]
        # cls_acc = pred/gt
        y_true, y_pred = gt_all.numpy().astype(np.uint8), pred_all.numpy().astype(np.uint8)
        test_accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        p = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        r = metrics.recall_score(y_true, y_pred, average='macro')

    l.info(f"round: {round+1}, Test Accuracy: {test_accuracy}, f1:{f1}, precision:{p}, recall:{r}")
    return best_model1, best_model2

@torch.no_grad()
def selection_cps_kmeans(model1, model2, selection_loader, n, round, args):
    model1.eval()
    model2.eval()

    # embeddings = torch.zeros((1, 2048))
    embeddings = torch.zeros((1, 128))
    logits1_val = torch.zeros((1, args.num_class))
    logits2_val = torch.zeros((1, args.num_class))
    names = ['']
    with torch.no_grad():
        for sample in selection_loader:
            # x1 = sample['img1'].cuda()
            # x2 = sample['img1'].cuda()
            x1 = sample['img'].cuda()
            x2 = sample['img'].cuda()
            label = sample['label']
            batch_names = sample['img_name']
            # logits1 = model1(x_s_batch)
            logits1, embeddings_cur1 = model1(x1, True, True)
            logits1 = torch.softmax(logits1, dim=1)
            logits1_val = torch.cat([logits1_val, logits1.detach().cpu()], dim=0)
            # logits2, embeddings_cur = model2(x2, True)
            logits2, embeddings_cur2 = model2(x2, True, True)
            logits2 = torch.softmax(logits2, dim=1)
            logits2_val = torch.cat([logits2_val, logits2.detach().cpu()], dim=0)
            names += batch_names
            embeddings_cur = (embeddings_cur1 + embeddings_cur2)/2
            embeddings = torch.cat([embeddings, embeddings_cur.detach().cpu()], dim=0)
    logits1_val = logits1_val[1:]
    logits2_val = logits2_val[1:]
    embeddings = embeddings[1:]
    names = names[1:]

    # compute decision boundary
    logits1_hard = torch.argmax(logits1_val, dim=1)
    logits2_hard = torch.argmax(logits2_val, dim=1)
    agree = (logits1_hard==logits2_hard).long()
    num_disagree = len(agree)-agree.sum()
    q_idx = agree.sort()[1][:num_disagree]
    # q_idx = agree.sort()[1][:4*n]
    names = [names[i] for i in q_idx]
    embeddings = embeddings[q_idx]

    # L1 loss

    # then Kmeans++
    cluster_learner = KMeans(n_clusters=n, init='k-means++', n_init='auto')
    cluster_learner.fit(embeddings)
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeddings - centers)**2
    # print(embeddings.shape, centers.shape)
    # print(cluster_idxs.shape)
    dis = dis.sum(axis=1)
    # print(dis.shape)
    q_idx = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

    return [names[i] for i in q_idx]

def main():
    args = get_arguments()
    seed_torch(args.seed)
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler('runs/al_ssl_cps_simclr.log', mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)
    l.info(f'seed:{args.seed}')
    # set gpu
    torch.cuda.set_device(args.gpu[0])
    # load dataset
    train_files = get_files('/home/ubuntu/data/lanfz/codes/tumor_AL_major/data_csv/crc100k-train.csv')
    np.random.shuffle(train_files)
    train_all_files = train_files.copy()
    train_files = train_files[:args.query_num]
    val_files = get_files('/home/ubuntu/data/lanfz/codes/tumor_AL_major/data_csv/crc100k-val.csv')
    test_files = get_files('/home/ubuntu/data/lanfz/codes/tumor_AL_major/data_csv/crc100k-test.csv')
    np.random.shuffle(train_files)
    print(f'train set len:{len(train_files)}')

    val_dataset = Tumor_dataset_val_cls(args, files=val_files)
    test_dataset = Tumor_dataset_val_cls(args, files=test_files)
    validation_loader = get_loader(args, val_dataset)
    test_loader = get_loader(args, test_dataset)

    # get labeled_idx and unlabeled idx
    labeled_num = len(train_files)
    all_data_num = len(train_all_files)

    labeled_data_img_names = []
    for i in range(labeled_num):
        img_path = train_files[i]['img']
        img_name = img_path.split('/')
        img_name = img_name[-1]
        labeled_data_img_names.append(img_name)

    labeled_idxs = []
    unlabeled_idxs = []
    for i in range(all_data_num):
        img_path = train_all_files[i]['img']
        img_name = img_path.split('/')
        img_name = img_name[-1]
        if img_name in labeled_data_img_names:
            labeled_idxs.append(i)
        else:
            unlabeled_idxs.append(i)
    
    # l.info(f'labeled:{len(labeled_idxs)},unlabeled:{len(unlabeled_idxs)}')
    print(f'labeled:{len(labeled_idxs)},unlabeled:{len(unlabeled_idxs)}')

    train_all_ds = Tumor_dataset_two_weak(args, files=train_all_files)
    train_loader = get_train_loader_ssl(args, train_all_ds, labeled_idxs, unlabeled_idxs)

    train_all_names = [list(item.values())[0] for item in train_all_files]
    train_names = [list(item.values())[0] for item in train_files]

    # here, AL settings
    n_rounds = args.rounds
    n_query = args.query_num
    l.info(f'n_query:{n_query}')

    for round in range(n_rounds):
        print("round:", round+1, 'training pool:', len(train_files))
        # get cur_labeled poor number
        train_num = [0]*args.num_class
        for item in train_names:
            train_num[train_all_files[train_all_names.index(item)]['label']] += 1
        print('train_num:', train_num)

        # get the model after training
        t0 = time.time()
        model1, model2 = train(train_loader, validation_loader, test_loader, l, round, args)
        t1 = time.time()
        print(t1-t0)

        # sample selection
        selection_names = list(set(train_all_names)-set(train_names))
        query_idx = [train_all_names.index(item) for item in selection_names]
        selection_files = [train_all_files[i] for i in query_idx]
        selection_loader = Tumor_dataset_val_cls(args, files=selection_files)
        # selection_loader = Tumor_dataset_two_weak(args, files=selection_files)
        selection_loader = get_loader(args, ds=selection_loader)

        q_names = selection_cps_kmeans(model1, model2, selection_loader, n_query, round, args)

        # update train_loader
        train_names += q_names
        train_names = list(set(train_names))
        train_idx =  [train_all_names.index(item) for item in train_names]
        train_files = [train_all_files[i] for i in train_idx]
        labeled_num = len(train_files)
        all_data_num = len(train_all_files)

        labeled_data_img_names = []
        for i in range(labeled_num):
            img_path = train_files[i]['img']
            img_name = img_path.split('/')
            img_name = img_name[-1]
            labeled_data_img_names.append(img_name)

        labeled_idxs = []
        unlabeled_idxs = []
        for i in range(all_data_num):
            img_path = train_all_files[i]['img']
            img_name = img_path.split('/')
            img_name = img_name[-1]
            if img_name in labeled_data_img_names:
                labeled_idxs.append(i)
            else:
                unlabeled_idxs.append(i)

        train_loader = get_train_loader_ssl(args, train_all_ds, labeled_idxs, unlabeled_idxs)


if __name__ == '__main__':
    main()
