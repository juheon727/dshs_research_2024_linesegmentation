import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import json
import multiprocessing

def process_mask(mask_path, img_resolution, mask_colors):
    img = cv2.imread(mask_path)
    img = cv2.resize(img, (img_resolution, img_resolution), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    mask = np.zeros((len(mask_colors), img_resolution, img_resolution), dtype=np.float32)
    for idx, (color, label) in enumerate(mask_colors.items()):
        mask[idx][np.all(img == color, axis=-1)] = 1
    return mask

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_resolution=448, img_transforms=None, task="train", verbose=True):
        self.path = path
        assert len(os.listdir(os.path.join(self.path, task, 'images'))) == len(os.listdir(os.path.join(self.path, task, 'masks')))
        self.img_transforms = img_transforms
        self.img_resolution = img_resolution
        self.task = task
        self.mask_colors = json.load(open(os.path.join(self.path, 'mask_colors.json')))
        self.mask_colors = {tuple(v) : i for i, (k, v) in enumerate(self.mask_colors.items())}

        mask_paths = [os.path.join(self.path, self.task, 'masks', '{:04d}.png'.format(i)) for i in range(len(os.listdir(os.path.join(self.path, self.task, 'masks'))))]

        # Parallel processing of masks using multiprocessing with guaranteed order
        with multiprocessing.Pool() as pool:
            if verbose:
                self.masks = list(tqdm(pool.starmap(process_mask, [(path, self.img_resolution, self.mask_colors) for path in mask_paths]), total=len(mask_paths), desc='Preprocessing Masks'))
            else:
                self.masks = list(pool.starmap(process_mask, [(path, self.img_resolution, self.mask_colors) for path in mask_paths]))

        self.masks = np.stack(self.masks)

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, self.task, 'images')))
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.task, 'images/{:04d}.jpg'.format(idx))).convert('RGB')
        img = img.resize((self.img_resolution, self.img_resolution))

        mask = torch.tensor(self.masks[idx])
        '''mask_img = cv2.imread(os.path.join(self.path, self.task, 'masks/{:04d}.png'.format(idx)))
        mask_img = cv2.resize(mask_img, (self.img_resolution, self.img_resolution), interpolation=cv2.INTER_NEAREST)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        mask_img = np.array(mask_img)
        mask = np.zeros((len(self.mask_colors), self.img_resolution, self.img_resolution), dtype=np.float32)
        for idx, (color, label) in enumerate(self.mask_colors.items()):
            mask[idx][np.all(img == color, axis=-1)] = 1'''

        if self.img_transforms is not None:
            img = self.img_transforms(img)
            #img = img.half()

        return img, mask

if __name__ == '__main__':
    config = json.load(open('/app/unet/config.json'))
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = Dataset(path=config['data_dir'], img_resolution=config['img_resolution'], img_transforms=img_transforms)
    img, mask = dataset.__getitem__(10)
    print(torch.sum(mask[1]))