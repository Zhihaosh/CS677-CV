import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# queryImage
img1 = cv.imread('./data/src_1.jpg',0)         
img2 = cv.imread('./data/dst_1.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# get number of features of these two images
print('source image features:' + str(len(kp1)))
print('target image features:' + str(len(kp2)))

# show the detected features overlaid on the images
img4 = cv.imread('./data/src_1.jpg',0)
img5 = cv.imread('./data/dst_1.jpg',0)
img4 = cv.drawKeypoints(img1,kp1,img4, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv.drawKeypoints(img2,kp2,img5, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('detected features overlaid on source images',img4)
cv.imshow('detected features overlaid on images',img5)

# create BFMatcher object
bf = cv.BFMatcher()

# Match descriptors which returns k of the best matches.
matches = bf.knnMatch(des1,des2,k=2)

matches_withoutlist = [] # for the question 2

# Apply ratio test
tmp = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        tmp.append([m])
        matches_withoutlist.append(m)

matches = tmp

# get number of matches
print("number of matches:" + str(len(matches)))

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x[0].distance)
matches_withoutlist = sorted(matches_withoutlist, key = lambda x:x.distance)

# Draw the top 20 scoring matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches[:20], None, flags=2)
plt.imshow(img3),plt.show()



# get keypoints of matches in both image
src_kp = []
tar_kp = [] 
for m in matches:
    src_kp.append(kp1[m[0].queryIdx].pt)
    tar_kp.append(kp2[m[0].trainIdx].pt)
#change the type tp np array
src_kp = np.asarray(src_kp)
tar_kp = np.asarray(tar_kp)
#find the homography
matrix, mask = cv.findHomography(src_kp, tar_kp, cv.RANSAC, 5.0)

print('the total numbers consistent:' + str(np.sum(mask)))

print('the computed homography matrix:')
print(matrix)

#find the boundary of the img1
h,w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

#do the transform to get the boundary for the src1 in img2.
dst = cv.perspectiveTransform(pts,matrix)

#set how to draw img2 with boundary dst
img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
matchesMask = mask.ravel().tolist()


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    matchesMask = matchesMask,
                   flags = 2)

#Show the top 10 (or more) matches that are found after homography has been computed
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches_withoutlist, None, **draw_params) # show top 20 mathches
plt.imshow(img3, 'gray'),plt.show()
