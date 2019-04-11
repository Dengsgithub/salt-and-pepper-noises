# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
import skimage
from noise_model import get_noise_model
import pre2 as pre2
from skimage import img_as_ubyte


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='./dataset/bijiao/zao',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="xin",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default='./impulse_clean/weights.hdf5',
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="impulse,70,70",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default='jieguo/set3/result',
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
#    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    image_yuan = cv2.imread("./dataset/black.png")
    noise = skimage.util.random_noise(image_yuan, mode='s&p', seed=None, clip=True, amount=0.5)
    noise = img_as_ubyte(noise)
    image = noise
    h, w, _ = image.shape
    image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
    h, w, _ = image.shape
    noise_image = image
    pred = model.predict(np.expand_dims(noise_image, 0))
    denoised_image = get_image(pred[0])
    cv2.imwrite("./result/see.png", denoised_image)


    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    # for image_path in image_paths:
    #     image = cv2.imread(str(image_path))
    #     image = pre2.yuchuli(image)
    #     h, w, _ = image.shape
    #     image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
    #     h, w, _ = image.shape
    #     noise_image = image
    #     pred = model.predict(np.expand_dims(noise_image, 0))
    #     denoised_image = get_image(pred[0])
    #     # out_image[:, :w] = image
    #     # out_image[:, w:w * 2] = noise_image
    #     # out_image[:, w * 2:] = denoised_image
    #
    #     if args.output_dir:
    #         #cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", image)
    #         cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", denoised_image)
    #     else:
    #         cv2.imshow("result", denoised_image)
    #         key = cv2.waitKey(-1)
    #         # "q": quit
    #         if key == 113:
    #             return 0


if __name__ == '__main__':
    main()