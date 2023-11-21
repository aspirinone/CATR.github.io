import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms
from config import cfg
import pdb
from misc import nested_tensor_from_videos_list
import torch
import random
import torch.nn.functional as F

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel

class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train'):
        super(S4Dataset, self).__init__()
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        im_mean = (124, 116, 104)
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform_strong = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5,1.0), shear=10),
            transforms.Resize(224, Image.BILINEAR),
            transforms.RandomCrop((224, 224), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform_strong = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5,1.0), shear=10),
            transforms.Resize(224, Image.NEAREST),
            transforms.RandomCrop((224, 224), pad_if_needed=True, fill=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.collator = Collator_train()
        self.collator_test = Collator_test()



    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        imgs, masks = [], []
        sequence_seed = np.random.randint(2147483647)
        if self.split == 'train': 
            for img_id in range(1, 6):
                reseed(sequence_seed)
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform_strong)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):
                reseed(sequence_seed)
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform_strong, mode='1')
                masks.append(mask)
        else:
            for img_id in range(1, 6):
                reseed(sequence_seed)
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):
                reseed(sequence_seed)
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
                masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0).squeeze(1).byte() 
        imgs_tensor = F.interpolate(imgs_tensor, size=(448,448), mode="bilinear", align_corners=False)
        masks_tensor = masks_tensor.numpy().astype(np.bool)
        referred_instance_idx = torch.tensor(0)

        # create the target dict for the center frame:
        target = {'masks':torch.tensor(masks_tensor),
                  'orig_size': masks_tensor.shape[-2:],  
                  'size': torch.tensor(masks_tensor.shape[-2:]),
                  'referred_instance_idx': referred_instance_idx,  
                  'iscrowd': torch.zeros(len(masks_tensor)), 
                  'image_id': video_name}

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, target, referred_instance_idx
        else:
            return imgs_tensor, audio_log_mel, target, category, video_name, referred_instance_idx #[5,3,224,224] [5,1,96,64] [5,1,224,224]
       

    def __len__(self):
        return len(self.df_split)

class Collator_train:
    def __call__(self, batch):
        imgs_tensor, audio_log_mel, target, referred_instance_idx = list(zip(*batch))
        device = imgs_tensor[0].device
        imgs_tensor = nested_tensor_from_videos_list(imgs_tensor)  # [T, B, C, H, W]
        audio_log = torch.stack(audio_log_mel,0)
        b, t, c, h, w = audio_log.shape
        audio_pad = torch.zeros((b, t), dtype=torch.bool, device=device)
        referred_instance_idx = referred_instance_idx[0]
        
        return imgs_tensor, audio_log, audio_pad, target,  referred_instance_idx # [5,4,3,224,224] mask [4,1,1,224,224]

class Collator_test:
    def __call__(self, batch):
        imgs_tensor, audio_log_mel, target, category, video_name, referred_instance_idx = list(zip(*batch))
        device = imgs_tensor[0].device
        imgs_tensor = nested_tensor_from_videos_list(imgs_tensor)  # [T, B, C, H, W]
        audio_log = torch.stack(audio_log_mel,0)
        b, t, c, h, w = audio_log.shape
        audio_pad = torch.zeros((b, t), dtype=torch.bool, device=device)
        category = category[0]
        video_name = video_name[0]
        referred_instance_idx = referred_instance_idx[0]
        return imgs_tensor, audio_log, audio_pad, target, category, video_name, referred_instance_idx


if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
