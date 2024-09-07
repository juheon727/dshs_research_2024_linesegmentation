import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import json
import os
from PIL import Image
import numpy as np
from dataset import Dataset
from unet import UNet

import torch

def intersection_over_union(y_hat: torch.Tensor, labels: torch.Tensor, n_classes: int):
    '''
    Calculates the Intersection over Union (IoU) between the prediction and labels.
    
    y_hat: Tensor of shape (N, C, H, W) where C is the number of classes, and H, W are the height and width of the image.
    labels: Tensor of shape (N, C, H, W) with ground truth labels (class indices).
    n_classes: Number of classes in the dataset.
    
    Returns:
        iou: List of IoU scores for each class.
    '''
    # Flatten the tensors to simplify computations
    y_hat = torch.argmax(y_hat, dim=1).view(-1)  # Predicted class labels
    labels = torch.argmax(labels, dim=1).view(-1)  # Ground truth class labels

    # Initialize lists to store the IoU values for each class
    iou = []

    for i in range(n_classes):
        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN) for each class
        intersection = ((y_hat == i) & (labels == i)).sum().item()
        union = ((y_hat == i) | (labels == i)).sum().item()

        if union == 0:
            # If union is zero, the IoU is undefined, set it to zero
            iou.append(0.0)
        else:
            # Calculate IoU
            iou.append(intersection / union)
    
    return iou


def evaluate(model):
    config = json.load(open('/app/unet/config.json'))
    mask_colors = json.load(open(os.path.join(config['data_dir'], 'mask_colors.json')))
    model.eval()
    img_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalization
    ])

    test_dataset = Dataset(
        path=config['data_dir'],
        img_resolution=config['img_resolution'],
        img_transforms=img_transforms,
        task='val',
        verbose=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    loss_f = nn.CrossEntropyLoss(reduce='mean')
    loss = 0
    iou = np.zeros((len(mask_colors)))
    for i, (img, label) in enumerate(test_dataloader):
        img, label = img.cuda(), label.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_hat = model(img)
            loss = loss_f(y_hat, label)
            loss += loss/len(test_dataloader)

            iou += np.array(intersection_over_union(y_hat, label, n_classes=len(mask_colors)))/len(test_dataloader)
        
        del y_hat, img, label

    return loss, iou
        
def overlay_mask(image: Image.Image, mask: torch.Tensor, mask_colors: dict):
    """
    Overlay the predicted mask on the original image.

    image: PIL Image of the original image.
    mask: Tensor of shape (H, W) containing the predicted class for each pixel.
    mask_colors: Dictionary with class names as keys and [R, G, B] as values.

    Returns:
        PIL Image with the mask overlay.
    """
    mask = mask.cpu().numpy()  # Convert mask to numpy array
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Map each class index to the corresponding RGB color
    for class_name, color in mask_colors.items():
        mask_rgb[mask == list(mask_colors.keys()).index(class_name)] = color

    mask_pil = Image.fromarray(mask_rgb)

    # Blend the original image with the mask
    overlay = Image.blend(image.convert("RGBA"), mask_pil.convert("RGBA"), alpha=0.5)

    return overlay


if __name__ == '__main__':
    config = json.load(open('/app/unet/config.json'))
    mask_colors = json.load(open(os.path.join(config['data_dir'], 'mask_colors.json')))
    model = UNet().cuda()
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'unet_best.pt')))
    model.eval()

    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = Dataset(
        path=config['data_dir'],
        img_resolution=config['img_resolution'],
        img_transforms=img_transforms,
        task='test'
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers']
    )

    loss_f = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    test_loss = 0
    test_iou = np.zeros((len(mask_colors),))

    for i, (img, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        img, label = img.cuda(), label.cuda()
        original_img = test_dataset.__getitem__(i)[0]  # Load original image
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_hat = model(img)
            loss = loss_f(y_hat, label)
            non_background_mask = (torch.argmax(label, dim=1) != 0).float()
            weighted_loss = loss * (1 + (config['non_background_multiplier'] - 1) * non_background_mask)
            mean_weighted_loss = weighted_loss.mean()
            test_loss += mean_weighted_loss / len(test_dataloader)
            test_iou += np.array(intersection_over_union(y_hat, label, n_classes=len(mask_colors))) / len(test_dataloader)
        
        y_hat_rgb = torch.argmax(y_hat, dim=1).squeeze(0).cpu()  # Convert to class indices

        mask_overlay = overlay_mask(transforms.ToPILImage()(original_img), y_hat_rgb, mask_colors)
        
        # Save the overlaid image
        mask_overlay.save(os.path.join(config['prediction_dir'], f'{i}.png'))

        del y_hat, img, label

    print('Loss: {:.4f}'.format(test_loss))
    print('mIoU: {:.4f}'.format(np.mean(test_iou)))
