import cv2
import numpy as np
mush = cv2.imread('mushroom_h.png')
mush_bgr = cv2.imread('mus_bgr_h.png')
mush_hsv = cv2.imread('mus_hsv_h.png')
res_mushroom_h=np.hstack([mush,mush_bgr,mush_hsv])
sky = cv2.imread('sky_h.png')
sky_bgr = cv2.imread('sky_bgr_h.png')
sky_hsv = cv2.imread('sky_hsv_h.png')
res_sky_h=np.hstack([sky,sky_bgr,sky_hsv])
cv2.imwrite('res_sky_h.png', res_sky_h)
cv2.imwrite('res_mushroom_h.png', res_mushroom_h)