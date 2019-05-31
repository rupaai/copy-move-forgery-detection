import cv2 as cv
import numpy as np
from keras.layers import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy.misc import imsave
import  numpy  as  np
from keras.layers import *

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input

class PreProcessing:
    def take_input(self, filename):
        return cv.imread(filename)
    def make_grayScale(self, img):
        return cv.COLOR_RGB2GRAY(img)
    def divide_LL_blocks(self, img, L):
        m, n, d = img.shape
        # img_gray = self.make_grayScale(img)
        print(m, n, m-L+1*n-L+1)
        blocks = np.zeros(((m-L+1)*(n-L+1), L, L, 3))
        k = 0
        for i in range(m-L+1):
            for j in range(n-L+1):
                blocks[k] = img[i:i + L, j:j + L, :]
                k += 1
        return blocks
class NeuralNetowrk:
    def __init__(self):
        super()
        # store the number of points and radius
        # self.numPoints = numPoints
        # self.radius = radius
        # self.image = image
    def extract_features(self, img):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(32, 32, 3), padding='VALID'))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 2
        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
        model.add(AveragePooling2D(pool_size=(5, 5)))

        # set of FC => RELU layers
        model.add(Flatten())

        # getting the summary of the model (architecture)
        model.summary()
        # img = cv.imread('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_2.png')

        # img = image.load_img(img_path, target_size=(530, 700))

        img_data = image.img_to_array(img)
        print(img_data.shape)
        img_data = np.expand_dims(img_data, axis=0)
        print(img_data.shape)
        img_data = preprocess_input(img_data)
        print(img_data.shape)

        vgg_feature = model.predict(img_data)
        return vgg_feature
        # print the shape of the output (so from your architecture is clear will be (1, 128))
        # print shape
        # print(vgg_feature.shape)
        #
        # # print the numpy array output flatten layer
        # print(vgg_feature.shape)
    # def getNearestblocks(self):
    #     bf = cv.BFMatcher()
    #     # print()
    #     print(img.shape)
    #     matches = bf.knnMatch(des1, des2, k=1)
    # def describe(self, block, eps=1e-7):
    #     # compute the Local Binary Pattern representation
    #     # of the image, and then use the LBP representation
    #     # to build the histogram of patterns
    #     lbp = feature.local_binary_pattern(block, self.numPoints,
    #                                        self.radius, method="uniform")
    #     (hist, _) = np.histogram(lbp.ravel(),
    #                              bins=np.arange(0, self.numPoints + 3),
    #                              range=(0, self.numPoints + 2))
    #
    #     # normalize the histogram
    #     hist = hist.astype("float")
    #     hist /= (hist.sum() + eps)
    #     print(lbp.shape)
    #     print(hist)
    #     # return the histogram of Local Binary Patterns
    #     return hist

    def u_lbp_val(self, block, shift, P, L):
        u_val = 0
        for i in range(1, P):
            u_val = abs(block[((i-1)//L) - 1, ((i-1) % L)] - block[0, 0]) + abs(block[((i+1)//L) - 1, ((i+1) % L)] - block[(i//L) - 1, (i % L)])
        return shift * u_val

    def lbp_ri_u2(self, blocks, shift, P, L):
        lbp = np.zeros((blocks.shape[0], 1))
        for i in range(blocks.shape[0]):
            if self.u_lbp_val(blocks[i], shift, P, L)>2:
                sum = 0
                for j in range(P):
                    sum += blocks[i][(j//L) - 1, (j % L)]
                lbp[i] = shift * sum
            else:
                lbp[i] = P + 1
        return lbp
    def lbp_hist(self, lbp, m, n, bin, L):
        k = 0
        lbp_hist = np.zeros((m*n, 1))
        for b in range(bin):
            for i in range(m):
                for j in range(n):
                    if self.equal_or_not(lbp[i*L + j], b):
                        lbp_hist[b] = lbp_hist[b] + 1

    def equal_or_not(self, a, b):
        if a == b:
            return 1
        else:
            return 0



class BlockMatching:
    def __init__(self, blocks):
        self.blocks = blocks
    def lex_sort(self):
        # feat = LocalBinaryPattern(24, 8)
        feat_mat = []
        for i in range(len(self.blocks)):
            # feat_mat.append(feat.describe(self.blocks[i]))
            a = 0
        return feat_mat


def euclidean_dis(train, test):
    dis = 0
    for i in range(train.shape[0]):
        dis += (train[i] - test[i])**2
    dis = dis ** 0.5
    print(dis)
    return dis

def KNNClassifier(features, th):
    # pred = np.zeros(yTest.shape)
    dis = []
    rep = []
    for i in range(len(features)):
        dis.append([])
        dis[i].append([0, i])
        for j in range(len(features) - 1):
            # if j in rep:
            #     continue
            # if abs(i-j)<5:
            #     continue
            ed = euclidean_dis(features[i], features[i])
            print(features[i], features[j])
            if ed <= th:
                dis[i].append([ed, j])
                rep.append(j)
            if len(dis[i]) >= 4:
                break
    return dis
def sort_pixels(block):
    flat = block.reshape(-1)
    return sorted(flat)
def lexicographical_sort(dis_mat, blocks):
    arr = []
    for i in range(len(dis_mat)):
        arr.append([])
        for j in range(len(dis_mat[i])):
            sorted_block = sort_pixels(blocks[dis_mat[i][j][1]])
            arr[i].append(sorted_block)
    return arr

def match_blocks(sorted_pix):
    mark = []
    for i in range(len(sorted_pix)):
        for j in range(len(sorted_pix[i]) - 1):
            for k in range(j+1, len(sorted_pix[i])):
                if sorted_pix[i][j] == sorted_pix[i][k]:
                    mark.append([j, k])
img_path = 'D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_2.png'
pp = PreProcessing()
blocks = pp.divide_LL_blocks(cv.imread(img_path), 32)
print(blocks.shape)
nn = NeuralNetowrk()
features = []
# feat = nn.extract_features(img_path)
for i in range(len(blocks)):
    features.append(nn.extract_features(blocks[i]))
# print(features.shape)
dis_mat = KNNClassifier(features, 0.5)
print(dis_mat)
sorted_pix = lexicographical_sort(dis_mat, blocks)
print(len(sorted_pix))