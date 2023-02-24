
import glob
import numpy as np
from tifffile import imread


def CONSEP_Class(DATA,X):
#DATA=X
    Data = sorted(glob('C:/Users/es255022/hover_net/dataset/training_data/consep1/consep/train/*.npy'))
    # print(Data)
    Data1  = list(map(np.load,X))
    # print(Data1[0].shape)
    X=[x[...,(0,1,2)] for x in Data1]
    Y=[y[...,3]for y in Data1]
    Y1=[y[...,3]for y in Data1]
    cls_dict=[cls[...,4]for cls in Data1]
    print(len(X))
    print(len(Y))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,Y, cls_dict,n_channel

def PanNuke_Class(X,Data,masks):
    # Data = sorted(glob('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke_new/*.npy'))
    # # print(Data)
    # Data1 = list(map(np.load, Data))
    # print(len(Data1))
    # output_mask = np.zeros((2656, 256, 256))
    # masks = np.load('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke/fold_1/Fold 1/masks/fold1/masks.npy')
    # for i in range(len(masks)):
    #     output_mask[i] = np.sum(masks[i][:, :, :5], axis=2
    X = np.load('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke/fold_1/Fold 1/images/fold1/images.npy')
    Data = sorted(glob('C:/Users/es255022/OneDrive -Teradata/Desktop/Dataset/pannuke_new/*.npy'))
    Data1 = list(map(np.load, Data))
    masks = np.load(
            'C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke/fold_1/Fold 1/masks/fold1/masks.npy')
    CL3 = [[0 for x in range(65536)] for y in range(len(masks))]
    output_mask = np.zeros((len(masks), 256, 256))
    for i in range(len(masks)):
        output_mask[i] = np.sum(masks[i][:, :, :5], axis=2)
    for i in range(len(masks)):
        x = np.nonzero(masks[i])
        CL3[i] = x[2]

    print('class length', len(CL3))
    Y = [y[..., 3] for y in Data1]
    # print(len(Y))
    Y1 = [y[..., 3] for y in Data1]
    # print(len(Y1))

    Y3 = [[0 for x in range(65536)] for y in range(len(masks))]
    print(len(Y))
    for i in range(len(masks)):
        Y3[i] = output_mask[i].flatten()
        print('instance map length', len(Y3))
        my_cls = []
        for i in range(len(masks)):
            my_cls.append(dict(zip(Y3[i], CL3[i])))
            cls=(*my_cls,)
        n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
        return X,Y,cls,n_channel

def PanNuke(img,label):
    # PANNUKE
    # X = np.load('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke/fold_1/Fold 1/images/fold1/images.npy')
    X=np.load(img)
    # Y = np.load('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/pannuke/fold_1/Fold 1/masks/fold1/masks.npy')
    masks=np.load(label)
    masks= masks.astype(int)
    output_mask = np.zeros((len(masks), 256, 256))
    for i in range(len(masks)):
        output_mask[i] = np.sum(masks[i][:, :, :5], axis=2)
    # Y = [y[...,1] for y in Y]

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,output_mask,n_channel

def CryoNuSeg(img,label):
    X = sorted(glob(img))
    Y = sorted(glob(label))
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    # Y = [y[...,0] for y in Y]
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,Y,n_channel
def CoNSeP(img,label):
    # CONSEP PNG
    from cv2 import imread
    # X = sorted(glob('C:/Users/es255022/hover_net/dataset/training_data/consep/img/*.png'))
    # Y = sorted(glob('C:/Users/es255022/hover_net/dataset/training_data/consep/mask/*.png'))
    X=sorted(glob(img))
    Y=sorted(glob(label))
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    Y = [y[..., 0] for y in Y]
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,Y,n_channel
def MoNuSaC(img,label):
    # X = sorted(glob('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/MoNuSAC/Images/*.tif'))
    X=sorted(glob(img))
    Y=sorted(glob(label))
    # Y = sorted(glob('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/MoNuSAC/newmask/*.jpg'))
    from tifffile import imread
    X = list(map(imread, X))
    print('done')
    from cv2 import imread
    Y = list(map(imread, Y))
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,Y,n_channel
def Kumar(img, label):
    from tifffile import imread
    # X = sorted(glob('C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/kumar-20220701T064616Z-001/kumar/train/Images/*.tif'))
    X=sorted(glob(img))
    X = list(map(imread, X))
    from scipy.io import loadmat
    Y=sorted(glob(label))
    # Y = sorted(glob(
    #     'C:/Users/es255022/OneDrive - Teradata/Desktop/Dataset/kumar-20220701T064616Z-001/kumar/train/Images/Labels/*.mat')

    Y1 = list(map(loadmat, Y))
    # Y = [y[...,0] for y in Y]
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    return X,Y,n_channel
