import cv2

#Read image
img = cv2.imread('./Images/2007_000464.jpg',cv2.COLOR_BGR2LAB)

#Display image
dis = cv2.pyrMeanShiftFiltering( img, 50, 50, 1)  
                                     
#save the result
cv2.imwrite('./tmp.jpg', dis)

print(img.shape)