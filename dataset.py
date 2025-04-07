from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

img_size = 224
class MyDataset(Dataset):
    def __init__(self, path, transforms=False, phase='train'):
        self.transforms = transforms
        self.phase = phase
        self.transform_true = A.Compose([A.Resize(img_size, img_size),
                                             A.HorizontalFlip(p=0.5),
                                             A.Rotate(limit=30, p=0.5),
                                             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                                             A.RandomResizedCrop(size=(img_size,img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
                                             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                             ToTensorV2()],
                                             bbox_params=A.BboxParams(format='pascal_voc', label_fields=[], min_visibility=0.1))

        self.transform_false = A.Compose([A.Resize(img_size, img_size),
                                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          ToTensorV2()],
                                          bbox_params=A.BboxParams(format='pascal_voc', label_fields=[], min_visibility=0.1))

        self.images, self.annotations_list, self.bboxes_list = self.load_images(path)

    def load_images(self,path):
        img_list = []
        masks_list = []
        annotations_list = []
        bboxes_list = []
        for file in os.listdir(path):
            if file.endswith(".png"):
                img = cv2.imread(os.path.join(path, file))
                h, w ,_ = img.shape
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img_list.append(image_rgb)

                file_annotations, _ = os.path.splitext(file)
                label_list, bbox_list = self.parse_train_annotation(file_annotations, h, w)
                annotations_list.append(label_list)
                bboxes_list.append(bbox_list)

        return img_list, annotations_list, bboxes_list

    def parse_train_annotation(self, filename, h, w):
        label_regex = re.compile(r'Original label for object \d+ ".*?" : "(.*)"')
        bbox_regex = re.compile(r'Bounding box for object \d+ ".*?" .*? : \((\d+), (\d+)\) - \((\d+), (\d+)\)')
        bbox_list = []
        label_list = [0,0,0,0]
        classes = ['bike', 'cars', 'motorbikes', 'person']
        try:
            with open(f'data/{self.phase}_annotations_label/{filename}.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    match = label_regex.match(line)
                    if match:
                        label_text = match.group(1)
                        index = classes.index(label_text) if label_text in classes else 1
                        label_list[index] = 1.0
                        continue

                    match = bbox_regex.match(line)
                    if match:
                        xmin = float(match.group(1))
                        ymin = float(match.group(2))
                        xmax = float(match.group(3))
                        ymax = float(match.group(4))

                        xmin = np.clip(xmin, 0, w)
                        ymin = np.clip(ymin, 0, h)
                        xmax = np.clip(xmax, 0, w)
                        ymax = np.clip(ymax, 0, h)

                        bbox_list.append([xmin, ymin, xmax, ymax])
        except FileNotFoundError:
            print(f'Error opening data/{self.phase}_annotations_label/{filename}.txt')

        return label_list, bbox_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_rgb = self.images[idx]
        boxes = self.bboxes_list[idx]

        if self.phase == 'train' or self.phase == 'val':
            annotations_list =  self.annotations_list[idx]

            if self.transforms:
                transformed  = self.transform_true(image=img_rgb, bboxes=boxes)
                img_transformed = transformed['image']
                boxes_transformed = transformed['bboxes']
                if len(boxes_transformed) < 6:
                    max_object = 6 - len(boxes_transformed)
                    boxes_transformed = boxes_transformed + [[-1,-1,-1,-1]] * max_object

            else:
                transformed = self.transform_false(image=img_rgb, bboxes=boxes)
                img_transformed = transformed['image']
                boxes_transformed = transformed['bboxes']
                if len(boxes_transformed) < 6:
                    max_object = 6 - len(boxes_transformed)
                    boxes_transformed = boxes_transformed + [[-1,-1,-1,-1]] * max_object

            return (img_transformed,
                    torch.tensor(boxes_transformed, dtype=torch.float32),
                    torch.tensor(annotations_list, dtype=torch.float32))

        else:
            transformed = self.transform_false(image=img_rgb, bboxes=boxes)
            img_transformed = transformed['image']
            boxes_transformed = transformed['bboxes']
            if len(boxes_transformed) < 6:
                max_object = 6 - len(boxes_transformed)
                boxes_transformed = boxes_transformed + [[-1, -1, -1, -1]] * max_object

            return (img_transformed,
                    torch.tensor(boxes_transformed, dtype=torch.float32))