from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    ################################################################
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    ################################################################
    if opt.recover_loss_weight:
        p1, p3 = [], []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            if "hm_rel" in key:
                p1.append(value)
            else:
                p3.append(value)
        params = [
            {'params': p1,
             'lr': opt.lr / opt.refine_weight,
             'weight_decay': 0},
            {'params': p3,
             'lr': opt.lr,
             'weight_decay': 0},
        ]
        print(opt.lr / opt.refine_weight, opt.lr)
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print('Starting training...')
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if epoch > opt.save_epoch:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)

            if opt.recover_loss_weight or opt.recover_ho_loss_weight:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
