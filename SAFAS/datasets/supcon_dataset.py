import os

from PIL import Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from SAFAS.utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class FaceDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, txt=None):
        # self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.strip('\n')
            words = line.split()
            imgs.append(words[0])
        self.imgs = imgs
        self.root_dir = root_dir
        self.transform = transform
        self.face_width = 400

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image_dir = os.path.join(self.root_dir, self.imgs[idx])
        labels = int('live' in image_dir)

        pil_image = Image.open(image_dir)
        pil_image = pil_image.convert("RGB")

        tensor_images = self.transform(pil_image)
        return tensor_images, labels

    def generate_square_images(self, image, info, range_scale=3):
        points = np.array(info['points'])
        dist = lambda p1, p2: int(np.sqrt(((p1 - p2) ** 2).sum()))
        width = dist(points[0], points[1])
        # height = max(dist(points[1], points[4]), dist(points[0], points[3]))
        center = tuple(points[2])

        angle = math.degrees(math.atan((points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])))
        rect = (center, (int(width * range_scale), int(width * range_scale)), angle)
        img_rows = image.shape[0]
        img_cols = image.shape[1]

        round = 0
        initial_scale = range_scale
        scale = range_scale
        min_scale = (256 / self.face_width) * initial_scale + 0.3

        while True:
            if inside_rect(rect=rect, num_cols=img_cols, num_rows=img_rows):
                break

            if scale < min_scale:
                pad_size = 300
                image = np.array(tf.pad(PIL.Image.fromarray(image), pad_size, padding_mode='symmetric'))
                center = (center[0] + pad_size, center[1] + pad_size)
                rect = (center, (int(width * scale), int(width * scale)), angle)
                break

            scale = range_scale - round * 0.1
            rect = (center, (int(width * scale), int(width * scale)), angle)
            round += 1

        scaled_face_size = int(self.face_width * scale / initial_scale)
        image_square_cropped = crop_rotated_rectangle(image=image, rect=rect)
        # vis_rotcrop(image, image_square_cropped, rect, center)
        image_resized = cv2.resize(image_square_cropped, (scaled_face_size, scaled_face_size))
        return image_resized

    def get_single_image_x(self, image_dir):
        image, info = self.sample_image(image_dir)
        h_img, w_img = image.shape[0], image.shape[1]
        h_div_w = h_img / w_img
        image_x = cv2.resize(image, (self.face_width, int(h_div_w * self.face_width)))
        return image_x
