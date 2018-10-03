
from PIL import Image
import os
import random
import numbers
import pickle
import sys
import numpy as np
import re
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def view_bar(num, total, img_num):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%% totalNum: %d, validImg: %d' % (
        "="*rate_num, " "*(100-rate_num), rate_num, num, img_num)
    sys.stdout.write(r)
    sys.stdout.flush()

def is_image_file(filename):
    isImage = False
    for extension in IMG_EXTENSIONS:
        if filename.endswith(extension):
            isImage = True
    return isImage

def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def load_file(filename):
    assert os.path.exists('./gen')
    with open(filename, 'rb') as fp:
        file = pickle.load(fp)
    return file

def save_file(file, filename):
    if not os.path.exists('./gen'):
        os.makedirs('./gen')
    with open(filename, 'wb') as fp:
        pickle.dump(file, fp)

def pil_loader(path, ycbcr=False):
    img = Image.open(path).convert('RGB')
    if ycbcr:
        img = img.convert('YCbCr')
    return img

def down_sampler(img, scale, interpolation=Image.BICUBIC):
    w, h = img.size[0], img.size[1]
    return img.resize((w//scale, h//scale), interpolation)

def image_crop(img, scale):
    w, h = img.size[0], img.size[1]
    to_crop_x = w % scale
    to_crop_y = h % scale
    return img.crop((to_crop_x // 2, to_crop_y // 2,
        to_crop_x // 2 + w - to_crop_x, to_crop_y // 2 + h - to_crop_y))

def make_train_dataset(dir, patch_size, class_to_idx, dataset_ratio=1):
    ratio = 3.8
    images = []
    count = 0
    data_file = os.path.join('./gen', "ImageNet" + "-train") # what is the data_file use for?
    if not os.path.exists(data_file):
        for target in os.listdir(dir):
            view_bar(count, len(os.listdir(dir)), len(images))
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        img = pil_loader(path)
                        if img.size[0] <= patch_size*ratio or img.size[1] <= patch_size*ratio:
                            continue
                        item = (path, class_to_idx[target])
                        images.append(item)
            count = count + 1
        save_file(images, data_file)
    else:
        images = load_file(data_file)
    return images[0 : int(len(images)*dataset_ratio)]


def make_val_dataset(dir, match=''):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            m = re.search('(%s)+'%(match), fname)
            if m is not None and is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
    return images

def imagenet_transform(patch_size):
    transform = transforms.Compose([
                transforms.RandomResizedCrop(patch_size),
                transforms.RandomHorizontalFlip(),
            ])
    return transform

class ImageNet(data.Dataset):
    def __init__(self, data_dir, scale, patch_size, data_augmentation=False, split="train", dataset_ratio=1, transform=imagenet_transform, target_transform=None,
                 loader=pil_loader):
        if split=="train":
            root = os.path.join(data_dir, "imagenet/train")
            classes, class_to_idx = find_classes(root)
            imgs = make_train_dataset(root, patch_size, class_to_idx, dataset_ratio)
        elif split=="val":
            root = os.path.join(data_dir, "imagenet/val")
            imgs = make_val_dataset(root, match='HR')
        elif split=='test':
            root = os.path.join(data_dir, "imagenet/test")
            imgs = make_val_dataset(root, match='')
        else:
            assert False, "invalid dataset split"
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.scale = scale
        # self.idx_scale = 0
        self.root = root
        self.split = split
        self.train = split =='train'
        self.imgs = imgs
        self.transform = transform(patch_size)
        self.target_transform = target_transform
        self.loader = loader
        self.ToTensor = transforms.ToTensor()


    def __getitem__(self, index, ycbcr=False):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
            # print('images shape', img.size)
        # hr_img = img
        if img.size[0] % self.scale == 0 and img.size[1] % self.scale == 0:
            # print('no image crop')
            hr_img = img
        else:
            hr_img = image_crop(img, self.scale)
        
        # down sampling
        lr_img = down_sampler(hr_img, self.scale, interpolation=Image.BICUBIC)
        
        # convert to tensor
        hr_img = self.ToTensor(hr_img)
        lr_img = self.ToTensor(lr_img)

        if ycbcr:
            hr_img = hr_img[0,:,:].view(1, hr_img.size(1), hr_img.size(2))
            lr_img = lr_img[0,:,:].view(1, lr_img.size(1), lr_img.size(2))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return lr_img, hr_img, path

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ImageNet test')
    parser.add_argument('--data_dir', type=str, default='/home/dengzeshuai/dataset/SR_training_datasets/imagenet',
                    help='dataset directory')
    parser.add_argument('--data_train', type=str, default='ImageNet',
                    help='train dataset name')
    parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
    parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
    parser.add_argument('--ycbcr', type=bool, default=False)
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    trainset = ImageNet(args)
    train_loader = MSDataLoader(
                args,
                trainset,
                batch_size=4,
                shuffle=True,
                pin_memory=not args.cpu
            )
    for i, (lr_img,  hr_img, path) in enumerate(train_loader):
        print(i, ' lr: ', lr_img.size(), ' hr: ', hr_img.size())
        if (i+1) % 10 == 0: 
            break;
