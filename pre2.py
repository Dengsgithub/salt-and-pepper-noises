import numpy as np

def yuchuli(img):
    h, w, _ = img.shape
    temp = []
    w_3 = 0
    w_5 = 0
    w_7 = 0
    for c in range(3):
        y = img[:, :, c]
        Y = img[:, :, c]
        for i in range(3, (h - 3) + 1):
            for j in range(3, (w - 3) + 1):
                tmp = y[i - 1:i + 2, j - 1:j + 2]
                w_3 = w_3+1
                maxnumber = np.max(tmp)
                minnumber = np.min(tmp)
                if Y[i, j] == maxnumber or Y[i, j] == minnumber:
                    tmp = tmp[tmp != maxnumber]
                    tmp = tmp[tmp != minnumber]
                    Len = len(tmp)
                    if Len != 0:
                        med = np.mean(tmp)
                        Y[i, j] = med
                    else:
                        tmp = y[i - 2:i + 3, j - 2:j + 3]
                        w_5 =w_5+1
                        maxnumber = np.max(tmp)
                        minnumber = np.min(tmp)
                        tmp = tmp[tmp != maxnumber]
                        tmp = tmp[tmp != minnumber]
                        Len = len(tmp)
                        if Len != 0:
                            med = np.mean(tmp)
                            Y[i, j] = med
                        else:
                            tmp = y[i - 3:i + 4, j - 3:j + 4]
                            w_7=w_7+1
                            maxnumber = np.max(tmp)
                            minnumber = np.min(tmp)
                            tmp = tmp[tmp != maxnumber]
                            tmp = tmp[tmp != minnumber]
                            Len = len(tmp)
                            if Len != 0:
                                med = np.mean(tmp)
                                Y[i, j] = med
                            else:
                                Y[i,j]=Y[i,j]

        Y[0, :] = Y[3, :]
        Y[1, :] = Y[3, :]
        Y[2, :] = Y[3, :]
        Y[h - 1, :] = Y[h - 4, :]
        Y[h - 2, :] = Y[h - 4, :]
        Y[h - 3, :] = Y[h - 4, :]
        Y[:, 0] = Y[:, 3]
        Y[:, 1] = Y[:, 3]
        Y[:, 2] = Y[:, 3]
        Y[:, w - 1] = Y[:, w - 4]
        Y[:, w - 2] = Y[:, w - 4]
        Y[:, w - 3] = Y[:, w - 4]
        print(w_7)
        print(w_5)
        print(w_3)
        temp.append(Y)
    result = np.hstack((temp[0].reshape(-1, 1), temp[1].reshape(-1, 1), temp[2].reshape(-1, 1)))
    result = result.reshape(h, w, 3)

    return result