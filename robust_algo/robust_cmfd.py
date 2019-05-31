import cv2
import numpy as np
from numpy.linalg import norm
import multiprocessing
def getFeatureVector(fn,q1,q2,q3,q4):

    img = cv2.imread(fn, 0)  # 1 chan, grayscale!
    imf = np.float32(img) / 255.0  # float conversion/scale
    dst = cv2.dct(imf)  # the dct
    img = np.uint8(dst) * 255.0
    return [sum(np.dot(q1, dst)), sum(np.dot(q2, dst)),sum(np.dot(q3, dst)),sum(np.dot(q4, dst))]
def getCircleMask(b):
    circle= np.zeros((b, b))
    for i in range(b):
        for j in range(b):
            if norm([i-(b+1)/2,j-(b+1)/2], 2) <= b/2:
                circle[i, j] = 1

def getFeatureMatrix(grayimage,B,FVsize):
    circle=getCircleMask(B)
    quarter1_mask=[[np.zeros(B/2,B/2),np.ones(B/2,B/2)], [np.zeros(B/2,B/2), np.zeros(B/2,B/2)]]
    quarter2_mask=[[np.ones(B/2,B/2) , np.zeros(B/2,B/2)], [np.zeros(B/2,B/2), np.zeros(B/2,B/2)]]
    quarter3_mask=[[np.zeros(B/2,B/2) , np.zeros(B/2,B/2)], [np.ones(B/2,B/2), np.zeros(B/2,B/2)]]
    quarter4_mask=[[np.zeros(B/2,B/2) , np.zeros(B/2,B/2)], [np.zeros(B/2,B/2), np.ones(B/2,B/2)]]

    q1= np.logical_and(quarter1_mask, circle)
    q2=np.logical_and(quarter2_mask, circle)
    q3=np.logical_and(quarter3_mask, circle)
    q4=np.logical_and(quarter4_mask, circle)
    area_quarter=sum(sum(q1))
    q1= q1/area_quarter
    q2= q2/area_quarter
    q3= q3/area_quarter
    q4= q4/area_quarter
    [M ,N]=grayimage.shape
    num_blocks=(M-B+1)*(N-B+1)
    FeatureMatrix= np.zeros(num_blocks,FVsize)
    Locations= np.zeros(num_blocks,2)
    rownum=0
    for x in range(N-B+1):
        for y in range(M-B+1):
            block=grayimage[y:y+B-1,x:x+B-1]
            rownum=rownum+1
        # %Store Features
            FeatureMatrix[rownum, :] = getFeatureVector(block, q1, q2, q3, q4)
            Locations[rownum, :] = [x, y]

# def func(sub_images, Mats, FVsize, B):
#     for h in range(4):
#         [Mats[:,:, h], locs[:,:, h]]=getFeatureMatrix(sub_images[:,:, h], B, FVsize);

def getFeatureMatrix_parallel(grayimage,B,FVsize):
    [M,N]= grayimage.shape
    c1=(M+B-1)/2;
    c2=(N+B-1)/2;
    sub_images = np.zeros(c1 ,c2, 4)
    sub_images[:,:,0]=grayimage[:c1 ,:c2]
    sub_images[:,:,1]=grayimage[:c1 ,c2-B+2:N]
    sub_images[:,:,2]=grayimage[c1-B+2:M, :c2]
    sub_images[:,:,3]=grayimage[c1-B+2:M ,c2-B+2:N]

    sections=[[0, 0], [0, c2-B+1], [ c1-B+1, 0], [c1-B+1, c2-B+1]]
    Mats= np.zeros((c1-B+1)*(c2-B+1),FVsize,4)
    locs= np.zeros((c1-B+1)*(c2-B+1),2,4)
    for h in range(4):
        [Mats[:, :, h], locs[:, :, h]] = getFeatureMatrix(sub_images[:, :, h], B, FVsize)

    FeatureMatrix=[Mats[:,:,0], Mats[:,:,2], Mats[:,:,3], Mats[:,:,4]]

    Locations=[locs(:,1,1)+sections(1,2),locs(:,2,1)+sections(1,1);...
                   locs(:,1,2)+sections(2,2),locs(:,2,2)+sections(2,1);...
                   locs(:,1,3)+sections(3,2),locs(:,2,3)+sections(3,1);...
                   locs(:,1,4)+sections(4,2),locs(:,2,4)+sections(4,1)];