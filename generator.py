from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir_zao,image_dir_yuan, source_noise_model, target_noise_model, batch_size=32, image_size=64):
        self.image_zao_paths = list(Path(image_dir_zao).glob("*.*"))
        self.image_yuan_paths = list(Path(image_dir_yuan).glob("*.*"))
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_zao_paths)
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            for i_image in range(self.image_num):
                aa = random.randint(0, self.image_num-1)
                image_zao_path = self.image_zao_paths[aa]
                image_yuan_path = self.image_yuan_paths[aa]
                image_zao = cv2.imread(str(image_zao_path))
                image_yuan = cv2.imread(str(image_yuan_path))
                h, w, _ = image_zao.shape

                if h >= image_size and w >= image_size:
                    h, w, _ = image_zao.shape
                    i = np.random.randint(h - image_size + 1)
                    j = np.random.randint(w - image_size + 1)
                    zao_patch = image_zao[i:i + image_size, j:j + image_size]
                    yuan_patch = image_yuan[i:i + image_size, j:j + image_size]
                    x[sample_id] = self.source_noise_model(zao_patch)
                    y[sample_id] = self.target_noise_model(yuan_patch)

                    sample_id += 1

                    if sample_id == batch_size:
                        return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir_zao, image_dir_yuan, val_noise_model):
        image_zao_paths = list(Path(image_dir_zao).glob("*.*"))
        image_yuan_paths = list(Path(image_dir_yuan).glob("*.*"))
        self.image_num = len(image_zao_paths)
        self.data = []

        for i_image in range(self.image_num):
            x = cv2.imread(str(image_zao_paths[i_image]))
            y = cv2.imread(str(image_yuan_paths[i_image]))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = x[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])


    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
