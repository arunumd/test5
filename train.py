import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from modeling import networks



class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)


        # Define network G and D (the output of Deeplab is score for each class, and the score is 
        # passed through softmax layer before going into PatchGAN)
        #================================== network ==============================================#
        network_G = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)
        
        softmax_layer = torch.nn.Softmax(dim=1)
        
        network_D = networks.define_D(24,
                                      64,
                                      netD='basic',
                                      n_layers_D=3,
                                      norm='batch',
                                      init_type='normal',
                                      init_gain=0.02,
                                      gpu_ids=self.args.gpu_ids)
        #=========================================================================================#



        train_params = [{'params': network_G.get_1x_lr_params(), 'lr': args.lr},
                        {'params': network_G.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        #================================== network ==============================================#
        optimizer_G = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer_D = torch.optim.Adam(network_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #=========================================================================================#
        
        
        # Define whether to use class balanced weights for criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        
        
        
        #=================== GAN criterion and Segmentation criterion ======================================#
        self.criterionGAN = networks.GANLoss('vanilla').to(args.gpu_ids[0])  ### set device manually
        self.criterionSeg = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        #===================================================================================================#
   
     
        self.network_G, self.softmax_layer, self.network_D = network_G, softmax_layer, network_D
        self.optimizer_G, self.optimizer_D = optimizer_G, optimizer_D

 
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.network_G = torch.nn.DataParallel(self.network_G, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.network_G)
            self.network_G = self.network_G.cuda()


        #====================== no resume ===================================================================#
        # Resuming checkpoint
        self.best_pred = 0.0
#        if args.resume is not None:
#            if not os.path.isfile(args.resume):
#                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
#            checkpoint = torch.load(args.resume)
#            args.start_epoch = checkpoint['epoch']
#            if args.cuda:
#                self.network_G.module.load_state_dict(checkpoint['state_dict'])
#            else:
#                self.network_G.load_state_dict(checkpoint['state_dict'])
#            if not args.ft:
#                self.optimizer.load_state_dict(checkpoint['optimizer'])
#            self.best_pred = checkpoint['best_pred']
#            print("=> loaded checkpoint '{}' (epoch {})"
#                  .format(args.resume, checkpoint['epoch']))
        #=======================================================================================================#
        
        

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        G_Seg_loss = 0.0
        G_GAN_loss = 0.0
        D_fake_loss = 0.0
        D_real_loss = 0.0
        
        #======================== train mode to set batch normalization =======================================#
        self.network_G.train()
        self.network_D.train()
        #======================================================================================================#
        
        
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer_G, i, epoch, self.best_pred)  # tune learning rate
            
            
            #================================= GAN training process (pix2pix) ============================================#
            
            # prepare tensors
            output_score = self.network_G(image)    # score map for each class in pixels
            output = self.softmax_layer(output_score)   # label for each pixel
            
            target_one_hot =  self.make_one_hot(target, C=21)   # change target to one-hot coding to feed into PatchGAN
            
            fake_AB = torch.cat((image, output), 1)
            real_AB = torch.cat((image, target_one_hot), 1)
            
            
            
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            
            # freeze G, unfreese D
            self.set_requires_grad(self.network_G, False)
            self.set_requires_grad(self.softmax_layer, False)
            self.set_requires_grad(self.network_D, True)
            
            # reset D grad
            self.optimizer_D.zero_grad()
            
            # fake input
            pred_fake = self.network_D(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # real input
            pred_real = self.network_D(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
            
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) / (2.0 * self.args.batch_size)
            
            loss_D.backward()
            self.optimizer_D.step() 
            
            
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            
            # unfreeze G, freese D
            self.set_requires_grad(self.network_G, True)
            self.set_requires_grad(self.softmax_layer, True)
            self.set_requires_grad(self.network_D, False)
            
            # reset G grad
            self.optimizer_G.zero_grad()
            
            # fake input should let D predict 1
            pred_fake = self.network_D(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            
            # Segmentation loss G(A) = B
            loss_G_CE = self.criterionSeg(output, target) * 1.0 # 1.0 is lambda_CE (weight for cross entropy loss)
            
            # combine loss and calculate gradients
            # lambda = 0.1
            loss_G = loss_G_GAN * 0.1 / self.args.batch_size + loss_G_CE
            loss_G.backward()
            
            self.optimizer_G.step()
            
            # display G and D loss
            G_Seg_loss += loss_G_CE.item()
            G_GAN_loss += loss_G_GAN.item()
            D_fake_loss += loss_D_fake.item()
            D_real_loss += loss_D_real.item()
            
            #===================================================================================================#
            
            
            
            tbar.set_description('G_Seg_loss: %.3f G_GAN_los: %.3f D_fake_loss: %.3f D_real_loss: %.3f' 
                                 % (G_Seg_loss / (i + 1), G_GAN_loss / (i + 1), D_fake_loss / (i + 1), D_real_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss_G_CE.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', G_Seg_loss, epoch)
        print('    [Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('    G Seg Loss: %.3f' % G_Seg_loss)

#======================================= no load checkpoint ==================#
#        if self.args.no_val:
#            # save checkpoint every epoch
#            is_best = False
#            self.saver.save_checkpoint({
#                'epoch': epoch + 1,
#                'state_dict': self.model.module.state_dict(),
#                'optimizer': self.optimizer.state_dict(),
#                'best_pred': self.best_pred,
#            }, is_best)
#=============================================================================#

    def validation(self, epoch):
        self.network_G.eval()
        self.evaluator.reset()
        
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.network_G(image)
            loss = self.criterionSeg(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        
        
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            
            #============== no save checkpoint ======================#
#            self.saver.save_checkpoint({
#                'epoch': epoch + 1,
#                'state_dict': self.model.module.state_dict(),
#                'optimizer': self.optimizer.state_dict(),
#                'best_pred': self.best_pred,
#            }, is_best)
            #=======================================================#
            
    #========================== new method ===============================# 
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def make_one_hot(self, labels, C=21):
        labels[labels==255] = 0.0
        labels = labels.unsqueeze(1)
        
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3), device=labels.device).zero_()
        target = one_hot.scatter_(1, labels.long(), 1.0)
            
        return target

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
