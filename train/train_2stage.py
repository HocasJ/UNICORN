import argparse
import collections
import os
import os.path as osp
import shutil
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.cuda import amp

from util.eval_metrics import extract_features_clip
from util.faiss_rerank import compute_jaccard_distance
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from ClusterContrast.cm import ClusterMemory, ClusterMemory_double,ClusterMemory2
from data.data_manager import process_query_part1_10, process_gallery_part1_10, process_query_sysu, process_gallery_sysu
# from data.data_manager import process_query_sysu, process_gallery_sysu
from data.dataloader import SYSUData_Stage2, IterLoader, TestData, MMMPData_Stage2
from util.eval import tester
from util.utils import IdentitySampler_nosk, GenIdx

from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR
# torch.backends.cuda.max_split_size_mb = 2750

from copy import deepcopy

def get_loader(dataset, batch_size, workers):
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return loader


def do_train_stage2(args,
                    dataset,
                    model,
                    optimizer,
                    scheduler,
                    loss_fn_rgb,
                    loss_fn_ir,
                    test_only = False
                    ):
    if test_only:
        start_time = time.monotonic()
        test_mode = [1, 2]
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        ])

        feat_dim = 2048
        if args.dataset == 'sysu':
            query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
        elif args.dataset == 'mmmp':
            query_img, query_label, query_cam = process_query_part1_10(args.data_path, mode=args.mode, exp_setting=args.exp_setting)
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

        # all test
        trial_nums = 1
        for trial in range(trial_nums):
            # print('-------test trial {}-------'.format(trial))
            if args.dataset == 'sysu':
                gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
            elif args.dataset == 'mmmp':
                gall_img, gall_label, gall_cam = process_gallery_part1_10(args.data_path, mode=args.mode, trial=trial, exp_setting=args.exp_setting)
            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

            cmc, mAP, mINP = tester(args, 0, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                    feat_dim=feat_dim,
                                    query_cam=query_cam, gall_cam=gall_cam)

            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP

        cmc = all_cmc / trial_nums
        mAP = all_mAP / trial_nums
        mINP = all_mINP / trial_nums
        print(
            "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                cmc[0], cmc[4],
                cmc[9], cmc[19],
                mAP, mINP))
        
        torch.cuda.empty_cache()
        end_time = time.monotonic()
        #print('Stage2 running time: ', timedelta(seconds=end_time - start_time))
        return
    



    best_acc = -1
    device = 'cuda'
    xent = SupConLoss(device)
    epochs = args.stage2_maxepochs
    start_time = time.monotonic()

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5)
    ])
    transform_train_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])


    batch = args.stage2_ims_per_batch
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir
    i_ter_rgb = num_classes_rgb // batch
    i_ter_ir = num_classes_ir // batch
    left_rgb = num_classes_rgb-batch* (num_classes_rgb//batch)
    left_ir = num_classes_ir-batch* (num_classes_ir//batch)
    if left_rgb != 0 :
        i_ter_rgb = i_ter_rgb+1
    if left_ir != 0 :
        i_ter_ir = i_ter_ir+1
    text_features_rgb = []
    text_features_ir = []
    label_list_rgb = []
    label_list_ir = []
    with torch.no_grad():
        for i in range(i_ter_rgb):
            if i+1 != i_ter_rgb:
                l_list_rgb = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_rgb = torch.arange(i*batch, num_classes_rgb)
            # with amp.autocast(enabled=True):
            text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
            text_features_rgb.append(text_feature_rgb.cpu())
            label_list_rgb.append(l_list_rgb.cpu())
        text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
        label_list_rgb  = torch.cat(label_list_rgb , 0).cuda()
        

    with torch.no_grad():
        for i in range(i_ter_ir):
            if i+1 != i_ter_ir:
                l_list_ir = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_ir = torch.arange(i*batch, num_classes_ir)
            # with amp.autocast(enabled=True):
            text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
            text_features_ir.append(text_feature_ir.cpu())
            label_list_ir.append(l_list_ir.cpu())

        text_features_ir = torch.cat(text_features_ir, 0).cuda()
        label_list_ir  = torch.cat(label_list_ir , 0).cuda()


    scaler = amp.GradScaler()
    losses = AverageMeter()
    losses_i2t = AverageMeter()
    losses_rgb = AverageMeter()
    losses_ir = AverageMeter()
    
    feat_dim = 2048

    # torch.cuda.empty_cache()

    for epoch in range(1, epochs+1):
        with torch.no_grad():
            
            print("==> Extract RGB features")
            dataset.rgb_cluster = True
            dataset.ir_cluster = False
            loader_rgb = get_loader(dataset, args.test_batch_size, args.workers)
            features_rgb, labels_rgb = extract_features_clip(model, loader_rgb, modal=1, get_image=False)
            features_rgb = torch.cat([features_rgb[path].unsqueeze(0) for path in dataset.train_color_path], 0).cuda()
            labels_rgb = torch.cat([labels_rgb[path].unsqueeze(0) for path in dataset.train_color_path], 0)

            print("==> Extract IR features")
            dataset.ir_cluster = True
            dataset.rgb_cluster = False
            loader_ir = get_loader(dataset, args.test_batch_size, args.workers)
            features_ir, labels_ir = extract_features_clip(model, loader_ir, modal=2, get_image=False)
            features_ir = torch.cat([features_ir[path].unsqueeze(0) for path in dataset.train_thermal_path], 0).cuda()
            labels_ir = torch.cat([labels_ir[path].unsqueeze(0) for path in dataset.train_thermal_path], 0)


            num_rgb = len(set(labels_rgb.tolist()))
            num_ir = len(set(labels_ir.tolist()))
            num_label = len(set(labels_ir.tolist()+labels_rgb.tolist()))
            print("num_label:", num_label)


        # generate new dataset
        @torch.no_grad()
        def generate_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            
            prototype_labels = torch.tensor(sorted(centers.keys())).to(device) #protype的label

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers, prototype_labels

        # m_features_rgb, prototype_labels_rgb = generate_features(labels_rgb.numpy(), features_rgb)
        # m_features_ir, prototype_labels_ir = generate_features(labels_ir.numpy(), features_ir)

        # labels_cat = torch.cat((labels_rgb, labels_ir), dim=0)
        # features_cat = torch.cat((features_rgb, features_ir), dim=0)
        # m_feature, prototye_lables = generate_features(labels_cat.numpy(), features_cat)


        # if epoch > 30:
        # m_features, prototype_labels = generate_features(labels_cat.numpy(), features_cat)

        # memory = ClusterMemory2(feat_dim, num_label, prototype_labels, temp=args.temp,
        #                     momentum=args.momentum, use_hard=args.use_hard).cuda()
        # memory.features = F.normalize(m_features, dim=1).cuda()
        
        # else: 
        m_features_rgb, prototype_labels_rgb = generate_features(labels_rgb.numpy(), features_rgb)
        m_features_ir, prototype_labels_ir = generate_features(labels_ir.numpy(), features_ir)


        memory = ClusterMemory_double(2048, num_rgb, num_ir, prototype_labels_rgb, prototype_labels_ir, temp=args.temp,
                            momentum=args.momentum, use_hard=args.use_hard, change_scale=1).cuda()
        
        memory.features_rgb = F.normalize(m_features_rgb, dim=1).cuda()
        memory.features_ir = F.normalize(m_features_ir, dim=1).cuda()



        del features_rgb, features_ir, loader_rgb, loader_ir

        # memory = ClusterMemory(2048, num_label, prototype_labels, temp=args.temp,
        #                        momentum=args.momentum, use_hard=args.use_hard).cuda()
        # memory.features = F.normalize(m_features, dim=1).cuda()


        end = time.time()
        trainset = MMMPData_Stage2(args, labels_rgb, labels_ir, transform_train_rgb, transform_train_ir)
        # trainset = SYSUData_Stage2(args.data_path, labels_rgb, labels_ir, transform_train_rgb, transform_train_ir)
    
        print("New Dataset Information---- ")
        print("  ----------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------")
        print("  visible  | {:5d} | {:8d}".format(len(np.unique(trainset.train_color_label)),
                                                  len(trainset.train_color_image)))
        print("  thermal  | {:5d} | {:8d}".format(len(np.unique(trainset.train_thermal_label)),
                                                  len(trainset.train_thermal_image)))
        print("  ----------------------------")
        print("Data loading time:\t {:.3f}".format(time.time() - end))

        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        sampler = IdentitySampler_nosk(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                                       args.num_instances, args.batch_size)

        trainset.cIndex = sampler.index1
        trainset.tIndex = sampler.index2

        trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                      num_workers=args.workers,
                                      drop_last=True)

        losses.reset() 
        losses_i2t.reset()
        losses_rgb.reset()
        losses_ir.reset()

        scheduler.step()
        model.train()
        for n_iter, (img_rgb, img_ir, target_rgb, target_ir, vid_rgb, vid_ir) in enumerate(trainloader):

            optimizer.zero_grad()
            img_rgb = img_rgb.to(device)
            target_rgb = target_rgb.to(device)
            vid_rgb = vid_rgb.to(device)
            
            img_ir = img_ir.to(device)
            target_ir = target_ir.to(device)
            vid_ir = vid_ir.to(device)
            


            # vids = torch.cat((vid_rgb, vid_ir), 0)

            # with amp.autocast(enabled=True):
            image_features, image_features_proj = model(x1=img_rgb, x2=img_ir, modal=0)
            

            logits_rgb = image_features[:img_rgb.size(0)] @ memory.features_rgb.t()
            logits_ir = image_features[img_rgb.size(0):] @ memory.features_ir.t()
         
            loss_id_rgb = loss_fn_rgb(logits_rgb, vid_rgb)
            loss_id_ir = loss_fn_ir(logits_ir, vid_ir)
            loss_id = loss_id_rgb + loss_id_ir


            out_rgb = image_features[:img_rgb.size(0)]
            out_ir = image_features[img_rgb.size(0):]

            loss_contrastive_rgb = xent(image_features_proj[:img_rgb.size(0)],  text_features_rgb, target_rgb, label_list_rgb)
            loss_contrastive_ir = xent(image_features_proj[img_rgb.size(0):],  text_features_ir, target_ir, label_list_ir)

            # loss = loss_contrastive_rgb + loss_contrastive_ir
            
            # if epoch >15:

            loss_contr, loss_contr_cross = memory(out_rgb, out_ir, target_rgb, target_ir)


            loss = loss_contr + loss_id + loss_contrastive_rgb + loss_contrastive_ir# +loss_i2t
            # loss = loss_contr + loss_contr_cross + loss_i2t#-v2

            # #+ loss_i2t 
            # if epoch > 5:
            #     loss = loss_contr_cross + loss_id#step2

            # if epoch >10:#step

            #     loss += loss_contr_cross

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            losses_i2t.update(loss_id.item())
            # losses_rgb.update(loss_rgb.item())
            # losses_ir.update(loss_ir.item())
            losses.update(loss.item())
            torch.cuda.synchronize()
            if n_iter % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}], losses_i2t: ({:.3f}), losses_rgb:({:.3f}), losses_ir:({:.3f}), total-losses:({:.3f}), Base Lr: {:.2e}"
                 .format(epoch, (n_iter + 1), len(trainloader),
                         losses_i2t.avg,  losses_rgb.avg, losses_ir.avg, losses.avg, scheduler.get_lr()[0]))
            
            

        if epoch >= 0:
            print('Test Epoch: {}'.format(epoch))
            test_mode = [1, 2]
            if args.dataset == 'sysu':
                query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
            elif args.dataset == 'mmmp':
                query_img, query_label, query_cam = process_query_part1_10(args.data_path, mode=args.mode, exp_setting=args.exp_setting)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

            # all test
            trial_nums = 1
            for trial in range(trial_nums):
                # print('-------test trial {}-------'.format(trial))
                if args.dataset == 'sysu':
                    gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                elif args.dataset == 'mmmp':
                    gall_img, gall_label, gall_cam = process_gallery_part1_10(args.data_path, mode=args.mode, trial=trial, exp_setting=args.exp_setting)
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                        feat_dim=feat_dim,
                                        query_cam=query_cam, gall_cam=gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP
                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

            cmc = all_cmc / trial_nums
            mAP = all_mAP / trial_nums
            mINP = all_mINP / trial_nums
            print(
                "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                    cmc[0], cmc[4],
                    cmc[9], cmc[19],
                    mAP, mINP))

            
            if cmc[0] > best_acc:
                best_acc = cmc[0]
                best_epoch = epoch
                best_mAP = mAP
                best_mINP = mINP
                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                if os.path.exists(args.model_path) is False:
                    os.mkdir(args.model_path)

                torch.save(state, os.path.join(args.model_path, args.logs_file + "0302_stage2.pth"))
            print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))

        torch.cuda.empty_cache()

    end_time = time.monotonic()
    print('Stage2 running time: ', timedelta(seconds=end_time - start_time))

