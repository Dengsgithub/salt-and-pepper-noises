import skimage.io as io
from skimage import img_as_ubyte, img_as_float
import skimage
import pre_process as pre
import numpy as np

def main():

    str1 = './dataset/BSDS300/*.jpg'
    imgTrain_o = io.ImageCollection(str1)
    img_train_o = []
    for i in range(len(imgTrain_o)):
        img_train_o.append(img_as_float(imgTrain_o[i]))

    for i in range(300):
        temp_n = skimage.util.random_noise(img_train_o[i], mode='s&p', seed=None, clip=True, amount=0.7)
        io.imsave('./dataset/train/val_yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i]))
        io.imsave('./dataset/train/val_zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))

    # for i in range(728):
    #     if i<91:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i],mode='s&p',seed = None,clip = True,amount = 0.2)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*1 and i<91*2:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*1],mode='s&p',seed = None,clip = True,amount = 0.3)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*1]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*2 and i<91*3:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*2],mode='s&p',seed = None,clip = True,amount = 0.4)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*2]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*3 and i<91*4:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*3],mode='s&p',seed = None,clip = True,amount = 0.5)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*3]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*4 and i<91*5:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*4],mode='s&p',seed = None,clip = True,amount = 0.6)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*4]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*5 and i<91*6:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*5],mode='s&p',seed = None,clip = True,amount = 0.7)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*5]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*6 and i<91*7:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*6],mode='s&p',seed = None,clip = True,amount = 0.8)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*6]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))
    #     elif i>=91*7 and i<91*8:
    #         img_train_o = get_image()
    #         temp_n = skimage.util.random_noise(img_train_o[i-91*7],mode='s&p',seed = None,clip = True,amount = 0.9)
    #         io.imsave('./dataset/train/yuan/'+np.str(i)+'.png',img_as_ubyte(img_train_o[i-91*7]))
    #         io.imsave('./dataset/train/zao/' + np.str(i) + '.png', img_as_ubyte(pre.yuchuli(temp_n)))



if __name__ == '__main__':
    main()