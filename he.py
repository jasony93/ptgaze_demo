import cv2
import numpy as np

img = cv2.imread('result/tmp/eye3.PNG',0)
he = cv2.equalizeHist(img)
res = np.hstack((img,he))
cv2.imshow('he', res)
cv2.waitKey(0)