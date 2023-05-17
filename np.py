import cv2
import numpy as np

img = np.full(shape=(64, 64, 3), fill_value=(255, 0, 0), dtype='uint8')

cv2.imwrite('blue.png', img)