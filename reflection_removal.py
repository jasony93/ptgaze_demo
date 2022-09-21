import cv2
import glob
from frr import FastReflectionRemoval
import numpy as np

globs = glob.glob(r'F:\DMS\pytorch_mpiigaze_demo\ptgaze\images\reflection\*')

img = cv2.imread(globs[1])
norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

alg = FastReflectionRemoval(h = 0.03)
dereflected_img = alg.remove_reflection(norm_image)

cv2.imshow('dereflected', dereflected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()