
import cv2
import numpy as np

rgb = np.random.randint(255, size=(900,800,3),dtype=np.uint8)
cv2.imshow('RGB',rgb)
cv2.waitKey(2)