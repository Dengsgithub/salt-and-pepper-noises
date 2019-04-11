# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
import skimage
from noise_model import get_noise_model
import pre2 as pre2
import skimage.io as io
import random


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='./dataset/new_data/train/set3zao/yuan',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="xin",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default='./impulse_clean/weights.100-1.890-26.49858.hdf5',
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="impulse,70,70",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default='./dataset/new_data/train/temp/yuan',
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        a = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        b = random.sample(a, 1)
        img = skimage.util.random_noise(image, mode='s&p', seed=None, clip=True, amount=b[0])
        img=img*[255]
        img = pre.yuchuli(img)
        img = pre2.yuchuli(image)


        if args.output_dir:
            print(img.dtype)
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", image)
        else:
            cv2.imshow("result", image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
