import cv2
import time
from frr import FastReflectionRemoval
import numpy as np

# cap = cv2.VideoCapture('result/20220706_095031.mp4')
cap = cv2.VideoCapture('result/glasses.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _gray = gray.copy()
    _, thresh_img = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    # inpainted = cv2.inpaint(gray, thresh_img, 40, cv2.INPAINT_TELEA)

    # blob detection
    # detector = cv2.SimpleBlobDetector()
    # blobs = detector.detect(gray)
    # im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    blurred = cv2.medianBlur(thresh_img, 5)
    # print(blurred.shape)
    contours,hierarchy = cv2.findContours(blurred,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    _contours = []
    contour_thresh = 10
    for c in contours:
        area = cv2.contourArea(c) #--- find the contour having biggest area ---
        # print(f'area: {area}')
        if(area < 200 and area > 20):
            _contours.append(c)
    
    _gray = cv2.drawContours(_gray, _contours, -1, (0,255,0), 3)


    detected_circles = cv2.HoughCircles(blurred,
                   cv2.HOUGH_GRADIENT, 1, 1, param1 = 100,
               param2 = 10, minRadius = 2, maxRadius = 20)

    # print(detected_circles)
    
    # Draw circles that are detected.
    if detected_circles is not None:

        # print('number of circles:', detected_circles.shape)
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv2.circle(gray, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(gray, (a, b), 1, (0, 0, 255), 3)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(gray)
    # mask2 = cv2.threshold(cl1 , 245, 255, cv2.THRESH_BINARY)[1]
    # clahe_inpainted = cv2.inpaint(cl1, mask2, 15, cv2.INPAINT_TELEA) 

    # norm_image = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # norm_image = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # alg = FastReflectionRemoval(h = 0.03)
    # dereflected_img = alg.remove_reflection(norm_image)
    # dereflected_img = np.float32(dereflected_img)
    # # print(dereflected_img.shape)
    # dereflected_gray = cv2.cvtColor(dereflected_img, cv2.COLOR_BGR2GRAY)

    # dereflected_gray = (dereflected_gray * 256).astype(np.uint8)

    # 80% reduction
    # reduced = np.array(gray, copy=True)
    # reduced[reduced > 250] -= 50

    # lab1 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # lab_planes1 = cv2.split(lab1)
    # lab_planes1 = clahe.apply(lab_planes1[0])
    # lab1 = cv2.merge(lab_planes1)
    # lab1 = lab1[:,0,:]
    # lab1 = cv2.rotate(lab1, cv2.ROTATE_90_CLOCKWISE)
    # lab1 = lab1[::-1]
    # print(lab1.shape)
    # clahe_gray = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    # clahe_gray = cv2.cvtColor(clahe_gray, cv2.COLOR_BGR2GRAY)

    combined = cv2.hconcat([gray, blurred])
    _combined = cv2.hconcat([combined, _gray])
    # __combined = cv2.hconcat([_combined, clahe_inpainted])
    # print(__combined.shape, dereflected_gray.shape)
    # print(__combined.dtype, dereflected_gray.dtype)
    # ___combined = cv2.hconcat([__combined, dereflected_gray])
    # ____combined = cv2.hconcat([___combined, reduced])
    
    # ___combined = cv2.hconcat([__combined, lab1])
    cv2.imshow('combined', _combined)
    # cv2.imshow('thresh', thresh_img)
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()