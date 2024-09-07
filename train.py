import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json
import os
from dataset import Dataset
from unet import UNet
from eval import evaluate

if __name__ == '__main__':
    config = json.load(open('/app/unet/config.json'))
    mask_colors = json.load(open(os.path.join(config['data_dir'], 'mask_colors.json')))
    model = UNet().cuda()
    # model.half()
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = Dataset(
        path=config['data_dir'],
        img_resolution=config['img_resolution'],
        img_transforms=img_transforms,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    scaler = torch.amp.GradScaler()

    iou_max = np.zeros((len(mask_colors),))
    epoch_max = 0
    loss_f = nn.CrossEntropyLoss(reduction='none')  # Set reduction to 'none' to get per-element loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0
        for img, label in train_dataloader:
            optimizer.zero_grad()
            img, label = img.cuda(), label.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_hat = model(img)
                
                # Calculate the raw loss
                loss = loss_f(y_hat, label.argmax(dim=1))

                # Create a mask for non-background class (assuming background is class 0)
                non_background_mask = (torch.argmax(label, dim=1) != 0).float()

                # Apply the multiplier for non-background classes
                weighted_loss = loss * (1 + (config['non_background_multiplier'] - 1)* non_background_mask)

                # Reduce the weighted loss to get the mean loss
                mean_weighted_loss = weighted_loss.mean()

                # Backpropagation and optimization steps
                scaler.scale(mean_weighted_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += mean_weighted_loss.item()

        val_loss, iou = evaluate(model)
        print(f"[{epoch}/{config['epochs']}] Train Loss: {epoch_loss / len(train_dataloader):.4f}, Val Loss: {val_loss}", end=', ')
        for i, (class_name, _) in enumerate(mask_colors.items()):
            print(f'{class_name} IoU: {iou[i]:.4f}', end=', ')
        print()

        if np.mean(iou) > np.mean(iou_max):
            iou_max = iou
            epoch_max = epoch
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f'unet_best.pt'))

    print(f"[{epoch_max}/{config['epochs']}] Maximum IoU: {iou_max}")
