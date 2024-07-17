import argparse
import os
import os.path as osp
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.backends import cudnn

from train.train_2stage import do_train_stage2
from util.loss.make_loss import make_loss
from train.train_1stage import do_train_stage1
from util.make_optimizer import make_optimizer_1stage, make_optimizer_2stage
from util.optim.lr_scheduler import WarmupMultiStepLR
from util.optim.scheduler_factory import create_scheduler
from data.dataloader import Unlabeld_SYSUData, Unlabeld_MMMPData
# from data_loader import Unlabeld_SYSUData_Pseudo
from model.make_model_clip import build_model

from util.utils import Logger

start_epoch = best_mAP = 0

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return cluster_loader


def main(args):
    # args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    cudnn.benchmark = True

    logs_time = args.logs_time
    logs_file = str(args.logs_file)

    sys.stdout = Logger(osp.join(args.logs_dir, logs_time, logs_file + '.txt'))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("==========\nArgs:{}\n==========".format(args))
    # Load datasets
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])
    end = time.time()
    print("==> Load unlabeled dataset")
    # data_path = '/home/cz/dataset/SYSU-MM01/'
    data_path = args.data_path
    if args.dataset == 'mmmp':
        # dataset = Unlabeld_SYSUData_Pseudo(data_dir=data_path, transform=transform_test, rgb_cluster=False,ir_cluster=False)
        # 
        dataset = Unlabeld_MMMPData(args, transform=transform_test, rgb_cluster=False,
                                        ir_cluster=False)
    elif args.dataset == 'sysu': 
        dataset = Unlabeld_SYSUData(args, transform=transform_test, rgb_cluster=False, ir_cluster=False)

    if -1 in dataset.train_color_label:
        n_color_class = len(np.unique(dataset.train_color_label)) - 1
    else:
        n_color_class = len(np.unique(dataset.train_color_label)) 
    if -1 in dataset.train_thermal_label:
        n_thermal_class = len(np.unique(dataset.train_thermal_label)) - 1
    else:
        n_thermal_class = len(np.unique(dataset.train_thermal_label))
    # num_classes = n_color_class + n_thermal_class

    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(n_color_class, len(dataset.train_color_image)))
    print("  thermal  | {:5d} | {:8d}".format(n_thermal_class, len(dataset.train_thermal_image)))
    print("  ----------------------------")
    # print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    # print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))

    # Create model
    model = build_model(args, n_color_class, n_thermal_class)
    model.cuda()

    # print('==> Resuming from checkpoint..')
    # if os.path.isfile("/root/part_processing/M3P-master/save/checkpoint_cctv_ir_cctv_rgb/test-cctv-ir-cctv-rgb0302_stage1.pth"):
    #     print('----load checkpoint-----')
    #     checkpoint = torch.load("/root/part_processing/M3P-master/save/checkpoint_cctv_ir_cctv_rgb/test-cctv-ir-cctv-rgb0302_stage1.pth")
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     print('no checkpoint found!')

    # Optimizer
    optimizer_1stage = make_optimizer_1stage(args, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=args.stage1_maxepochs, lr_min=args.stage1_lrmin,
                                           warmup_lr_init=args.stage1_warmup_lrinit, warmup_t=args.stage1_warmup_epoch, noise_range=None)
    
    # loss_func_rgb = make_loss(num_classes=n_color_class)
    # loss_func_ir = make_loss(num_classes=n_thermal_class)

    loss_func_rgb = make_loss(num_classes=n_color_class)
    loss_func_ir = make_loss(num_classes=n_thermal_class)

    # do_train_stage1(args, dataset, model, optimizer_1stage, scheduler_1stage)

    optimizer_2stage = make_optimizer_2stage(args, model)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, args.stage2_steps, args.stage2_gamma, args.stage2_warmup_factor,
                                         args.stage2_warmup_iters, args.stage2_warmup_method)
    
    if args.test_only:
        checkpoint = torch.load(args.load_stage2_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    do_train_stage2(args, dataset, model, optimizer_2stage, scheduler_2stage, loss_func_rgb, loss_func_ir, args.test_only)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrastive learning on unsupervised Cross re-ID")
    args_main = parser.parse_args()

    args = yaml.load(open('config/config_sysu.yaml'), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)
    print(torch.cuda.current_device())
    main(args)
