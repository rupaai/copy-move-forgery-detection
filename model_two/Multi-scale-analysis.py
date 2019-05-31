import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class MultiScalAnalysis:
    def __init__(self, img):
        self.img = img
    def pre_processing(self, img):
        img = cv.COLOR_RGB2HSV(img)
        m, n, _ = img.shape
        const = max(img) - min(img)
        for i in range(m):
            for j in range(n):
                img[i][j][0] = (img[i][j][0] - min(img)) / const
                img[i][j][1] = (img[i][j][1] - min(img)) / const
                img[i][j][2] = (img[i][j][2] - min(img)) / const
        return img
    def keyPoints_detection(self, img):
        img_clr = cv.imread(img)
        img1 = cv.imread(img, 0)
        # cv.xfeatures2d_SURF.setHessianThreshold(0)
        surf = cv.xfeatures2d.SURF_create(1000)
        kp1, des1 = surf.detectAndCompute(img1[:, :256], None)
        kp2, des2 = surf.detectAndCompute(img1[:, 256:], None)

        # #orb = cv.ORB_create()
        # # kp, des = orb.detectAndCompute(img1, None)
        img2 = img_clr.copy()
        # for marker in kp:
        #     img2 = cv.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(120, 200, 20))
        img_surf1 = cv.drawKeypoints(img2[:, :256, :], kp1, None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_surf2 = cv.drawKeypoints(img2[:, 256:, :], kp2, None,  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('keypoints', img_surf1)
        cv.imshow('sav', img_surf2)
        cv.waitKey(0)
        return kp1, des1, kp2, des2

    def nearest_neighbour(self, kp1, img1, des1, kp2, img2, des2):
        bf = cv.BFMatcher()
        # print()
        print(img.shape)
        matches = bf.knnMatch(des1, des2, k=2)
        # print(img.shape)
        # Apply ratio test
        good = []
        print(len(matches))
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append([m])
        #     for n in m:
        #         print(n.distance, n.imgIdx, n.trainIdx, n.queryIdx)
            # print()
            # print()
            # print()
        # print(len(kp))
        # i = 0
        # print(len(matches))
        # cv2.drawMatchesKnn expects list of lists as matches.
        for i in range(len(kp1)):
            print(kp1[i].pt)
            # print(kp2[matches[i]])
            for n in matches[i]:
                print(kp2[n.queryIdx].pt)
            # i += 1

        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        cv.imshow('adfa', img3)
        cv.waitKey(0)
        return good, img3

    def clustering_process(self, matches, kp1, kp2, th, n=4):
        gs = []
        gd = []
        for i in range(4):
            gs.append([])
            gd.append([])
        P = [[], [], [], []]
        ang = np.zeros((len(kp1)))
        print(ang.shape)
        x1 = y1 = 0
        for i in range(len(kp1)):
            x0, y0 = kp1[i].pt
            # print(kp2[matches[i]])
            for n in matches[i]:
                x1, y1 = kp2[n.queryIdx].pt
            m = (y1-y0)/(x1-x0)
            an = np.arctan(m)
            # print(an)
            ang[i] = (180 * an)/np.pi
            k = -1
            print(ang[i])
            if 0 <= ang[i] <= 89:
                k = 0
                print(k)
                print(len(gs[0]))
                if len(gs[0]) == 0:
                    gs[0].append([])
                    gd[0].append([])
                    gs[0][0].append(kp1[i].pt)
                    gd[0][0].append(kp2[n.queryIdx].pt)
                    # P[0].append([gs0, gd0])
                    # print(k)
            elif 90 <= ang[i] <= 179:
                k = 1

                if len(gs[1]) == 0:
                    gs[1].append([])
                    gd[1].append([])
                    gs[1][0].append(kp1[i].pt)
                    gd[1][0].append(kp2[n.queryIdx].pt)
                    # P[1].append([gs1, gd1])
            if 180 <= ang[i] <= 269:
                k = 2

                if len(gs[2]) == 0:
                    gs[2].append([])
                    gd[2].append([])
                    gs[2][0].append(kp1[i].pt)
                    # gs[1][0].append(kp1[i].pt)
                    gd[2][0].append(kp2[n.queryIdx].pt)
                    # P[2].append([gs2, gd2])
            if 270 <= ang[i] <= 359:
                k = 3

                if len(gs[3]) == 0:
                    gs[3].append([])
                    gd[3].append([])
                    gs[3][0].append(kp1[i].pt)
                    gd[3][0].append(kp2[n.queryIdx].pt)
                    # P[3].append([gs3, gd3])


            else:
                F = False
                for x in P[k]:
                    for m in x:
                        da = self.euclidean_dis(m[0], kp1[i].pt)
                        db = self.euclidean_dis(m[1], kp2[n.queryIdx].pt)
                    if da<th and db<th:
                        F = True
                    if F:
                        gs[k][len(gs[k])-1].append(kp1[i].pt)
                        gd[k][len(gs[k])-1].append(kp2[n.queryIdx].pt)
                        # P[k].append([kp1[i].pt, kp2[n.queryIdx].pt])
                if ~F:
                    gs[k].append([])
                    print(len(gs[k])-1)
                    gs[k][len(gs[k])-1].append(kp1[i].pt)
                    gd[k][len(gs[k])-1].append(kp2[n.queryIdx].pt)
                    # P[k].append([kp1[i].pt, kp2[n.queryIdx].pt])
                # print(len(gs))
                # print(k)
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)
        for i in range(len(gs[0])):
        # Create a Rectangle patch
            rect = patches.Rectangle((gs[0][i], gd[0][i]), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()
        return [gs, gd]

    def euclidean_dis(self, a, b):
        return (a**2 + b**2) ** 0.5
    def pyramidal_decomposition(self, img):
        # gray = cv.COLOR_RGB2GRAY(img)
        # dst = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
        # img1 = cv
        # pct = 0.5
        # newsize = (int(img.shape[0] * pct), int(img.shape[1] * pct))
        img1 = cv.resize(img, (256, 256))

        # newsize = (int(img1.shape[0] * pct), int(img1.shape[1] * pct))
        img2 = cv.resize(img1, (128, 128))

        # newsize = (int(img2.shape[0] * pct), int(img2.shape[1] * pct))
        img3 = cv.resize(img2, (64, 64))

        return img1, img2, img3
    # def
multi = MultiScalAnalysis('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png')
kp1, des1, kp2, des2 = multi.keyPoints_detection('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png')
img = cv.imread('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png')
img_gray = cv.imread('D:/Studies/4.2/DIP/Project/DIgital_Image_Processing/demo_1.png', 0)
match, img = multi.nearest_neighbour(kp1, img[:, :256, :], des1, kp2, img[:, 256:, :], des2)
img1, img2, img3 = multi.pyramidal_decomposition(img)
cv.imshow('img1', img1)
cv.waitKey(0)
cv.imshow('img2', img2)
cv.waitKey(0)
cv.imshow('img3', img3)
cv.waitKey(0)
# dst = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
# [gs, gd] = multi.clustering_process(match, kp1, kp2, 0.4)
print(len(kp1), len(kp2))