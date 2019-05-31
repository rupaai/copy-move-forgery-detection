import cv2
import numpy as np
FVsize=9
neighbor_radios=12
min_neighbors=6
max_std=10
B=24
search_th=50
distance_th=120
color=[255,0,0]
degree=4
r=15
p=1
q=8
rgb_img = cv2.imread('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png')
gray_img = cv2.imread('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png', 0)
m, n = gray_img.shape
num_blocks=(m-B+1)*(n-B+1)
SelMat = np.ones(m, n)
mar = B/2
SelMat[:mar,:]=0
SelMat[:,:mar] = 0
SelMat[m-mar:m,:] = 0
SelMat[:,n-mar:n]=0
