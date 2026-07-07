#coding=utf-8
"""RASC-Net training entry point.

Optimizes the joint objective (manuscript Eqs. 21-22):

    L_total = L_seg + alpha * L_dis + beta * L_cons

where
    L_seg  = Dice + weighted-CE on the fused prediction (+ auxiliary sep/prm heads),
    L_dis  = sum over scales {3,4} of lambda_ana L_ana + lambda_mod L_mod + lambda_rec L_rec,
    L_cons = subset consistency between two random modality subsets.
"""
import argparse
import os
import time
import logging
import random
import numpy as np

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from rascnet import RASC_Net
from rascnet.losses import DisentanglementObjective, subset_consistency_loss
from data.transforms import *  # noqa: F401,F403 (needed by eval(transforms))
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--datapath', default='BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str,
                    choices=['BRATS2020', 'BRATS2018', 'BRATS2015'])
parser.add_argument('--savepath', default='output/rascnet_brats2020', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--iter_per_epoch', default=-1, type=int,
                    help='-1 uses len(train_loader)')
parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--crop_size', default=112, type=int)

# ---- RASC-Net specific ----
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight of disentanglement loss L_dis (Eq. 22)')
parser.add_argument('--beta', default=0.4, type=float,
                    help='weight of subset consistency loss L_cons (Eq. 22)')
parser.add_argument('--lambda_ana', default=1.0, type=float)
parser.add_argument('--lambda_mod', default=1.0, type=float)
parser.add_argument('--lambda_rec', default=1.0, type=float)
parser.add_argument('--tau', default=0.1, type=float, help='contrastive temperature')
parser.add_argument('--use_reg_loss', action='store_true', default=True,
                    help='auxiliary per-modality segmentation regularizer')
parser.add_argument('--no_gating', action='store_true', default=False,
                    help='ablation: disable reliability-aware gating')
parser.add_argument('--no_fusion', action='store_true', default=False,
                    help='ablation: disable stability-constrained fusion')
parser.add_argument('--no_consistency', action='store_true', default=False,
                    help='ablation: disable subset consistency loss')
parser.add_argument('--no_disentangle_loss', action='store_true', default=False,
                    help='ablation: disable L_dis contrastive/recon losses')

args = parser.parse_args()
setup(args, 'training')
args.train_transforms = (
    'Compose([RandCrop3D(({0},{0},{0})), RandomRotion(10), '
    'RandomIntensityChange((0.1,0.1)), RandomFlip(0), '
    'NumpyType((np.float32, np.int64)),])'.format(args.crop_size)
)
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

# 15 missing-modality masks (order: FLAIR, T1ce, T1, T2)
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
mask_name = ['t2', 't1c', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    num_cls = 5 if args.dataname == 'BRATS2015' else 4

    model = RASC_Net(num_cls=num_cls,
                     use_gating=not args.no_gating,
                     use_fusion=not args.no_fusion)
    model = torch.nn.DataParallel(model).cuda()

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    dis_objective = DisentanglementObjective(
        lambda_ana=args.lambda_ana, lambda_mod=args.lambda_mod,
        lambda_rec=args.lambda_rec, tau=args.tau)

    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file, test_file = 'train.txt', 'test.txt'
    else:  # BRATS2018 uses split 3 by default
        train_file, test_file = 'train3.txt', 'test3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath,
                                  num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath,
                                      test_file=test_file)
    train_loader = MultiEpochsDataLoader(dataset=train_set, batch_size=args.batch_size,
                                         num_workers=8, pin_memory=True, shuffle=True,
                                         worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                        num_workers=0, pin_memory=True)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('########### evaluation across 15 modality combinations ###########')
            for i, mask in enumerate(masks):
                logging.info('{}'.format(mask_name[i]))
                dice_score = test_softmax(test_loader, model, dataname=args.dataname,
                                          feature_mask=mask, mask_name=mask_name[i],
                                          crop_size=args.crop_size)
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
        return

    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('############# training #############')
    iter_per_epoch = len(train_loader) if args.iter_per_epoch == -1 else args.iter_per_epoch
    train_iter = iter(train_loader)

    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            out = model(x, mask)
            fuse_pred = out['fuse_pred']
            sep_preds = out['sep_preds']
            prm_preds = out['prm_preds']

            # ---- segmentation losses (Eq. 21) ----
            fuse_cross = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross + fuse_dice

            sep_cross = torch.zeros(1).cuda().float()
            sep_dice = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross + sep_dice

            prm_cross = torch.zeros(1).cuda().float()
            prm_dice = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross + prm_dice

            use_reg = 1.0 if args.use_reg_loss else 0.0
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss * use_reg + prm_loss
            else:
                loss = fuse_loss + sep_loss * use_reg + prm_loss

            # ---- disentanglement objective L_dis (Eq. 11) ----
            if args.no_disentangle_loss:
                L_dis = torch.zeros(1).cuda().float()
                dis_logs = {}
            else:
                L_dis, dis_logs = dis_objective(out['anatomy_vec'], out['style_vec'],
                                                out['feat'], out['recon'])
                loss = loss + args.alpha * L_dis

            # ---- subset consistency L_cons (Eq. 19) ----
            if args.no_consistency:
                L_cons = torch.zeros(1).cuda().float()
            else:
                L_cons = subset_consistency_loss(out['z_s1'], out['z_s2'])
                loss = loss + args.beta * L_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice.item(), global_step=step)
            writer.add_scalar('L_dis', float(L_dis), global_step=step)
            writer.add_scalar('L_cons', float(L_cons), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format(
                epoch + 1, args.num_epochs, i + 1, iter_per_epoch, loss.item())
            msg += 'fuse:{:.4f}, sep:{:.4f}, prm:{:.4f}, '.format(
                fuse_loss.item(), sep_loss.item(), prm_loss.item())
            msg += 'L_dis:{:.4f}, L_cons:{:.4f}'.format(float(L_dis), float(L_cons))
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()}, file_name)
        if (epoch + 1) % 50 == 0 or epoch >= (args.num_epochs - 10):
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                       os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1)))

    logging.info('total time: {:.4f} hours'.format((time.time() - start) / 3600))

    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('########### final evaluation ###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(test_loader, model, dataname=args.dataname,
                                      feature_mask=mask, crop_size=args.crop_size)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))


if __name__ == '__main__':
    main()
