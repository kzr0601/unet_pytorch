import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')

dir_img = Path('../../unet_pretrain/Pytorch-UNet/data/imgs/')
dir_mask = Path('../../unet_pretrain/Pytorch-UNet/data/masks/')
valid_dir_img = Path('../../unet_pretrain/Pytorch-UNet/data/valid_imgs/')
valid_dir_mask = Path('../../unet_pretrain/Pytorch-UNet/data/valid_masks/')

modelpath = './checkpoints_base_changeloss/'
if not os.path.exists(modelpath):
    os.mkdir(modelpath)
dir_checkpoint = Path(modelpath)

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    # 1. Create dataset
    try:
        train_set = CarvanaDataset(dir_img, dir_mask, img_scale, True)
        val_set = CarvanaDataset(valid_dir_img, valid_dir_mask, img_scale, True)
    except (AssertionError, RuntimeError):
        train_set = BasicDataset(dir_img, dir_mask, img_scale, True)
        val_set = BasicDataset(valid_dir_img, valid_dir_mask, img_scale, True)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = len(os.listdir(str(dir_img)))
    n_val = len(os.listdir(str(valid_dir_img)))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    num_one, num_zero = 616476, 156285924
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() #weight=torch.FloatTensor([num_one, num_zero]).cuda())
    global_step = 0

    # 5. Begin training
    loss_array = []
    valid_acc_array = []
    max_val_score = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                    loss_array.append(loss)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                #if global_step % (n_train // (10 * batch_size)) == 0:   
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        val_score = evaluate(net, val_loader, device)
        valid_acc_array.append(val_score)
        scheduler.step(val_score)

        logging.info('Validation Dice score: {}'.format(val_score))
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation Dice': val_score,
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(true_masks[0].float().cpu()),
                'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })

        if(val_score > max_val_score):
            max_val_score = val_score
            torch.save(net.state_dict(), str(dir_checkpoint / 'bestmodel.pth'))
            logging.info(f'bestmodel: Checkpoint {epoch + 1} saved!')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

    return loss_array, valid_acc_array 

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    #params = torch.load('unet_carvana_scale0.5_epoch1.pth')
    #net.load_state_dict(params)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        loss_array, valid_acc_array = train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
        print(len(loss_array), len(valid_acc_array))
        x = [_ for _ in range(len(loss_array))]
        plt.figure(1)
        plt.plot(x, loss_array, 'b-')
        plt.savefig("loss.png")
        x = [_ for _ in range(len(valid_acc_array))]
        plt.figure(2)
        plt.plot(x, valid_acc_array, 'r-')
        plt.savefig("valid_acc.png")

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
