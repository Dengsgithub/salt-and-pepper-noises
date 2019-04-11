import skimage.io as io

import time as time
def main():
    x = io.imread("./dataset/70/0.png")
    start_time = time.time()
    y = pre.yuchuli(x)
    stop_time = time.time()
    print(stop_time-start_time)


if __name__ == '__main__':
    main()