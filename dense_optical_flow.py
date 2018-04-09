import cv2
import numpy as np


# Read in images and cut into 4 slices (mod so it does it for other games)
def img_cut():
    for i in range(3001):
        image_name = 'Data/Breakout/breakout' + str(i) + '.png'
        img = cv2.imread(image_name)

        if i == 3000:
            print image_name

        crop_height = 84
        crop_y = 0
        crop_width = 84
        crop_x = 0

        for i in range(4):
            frame = img[crop_y:crop_height, crop_x:crop_width]
            # print crop_x, " : ", crop_width
            crop_x += crop_height
            crop_width += crop_height
        # print "\n"
    return frame

img_cut()


prvs = cv2.cvtColor(img_cut(), cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(img_cut())
hsv[...,1] = 255
# while(1):
next = cv2.cvtColor(img_cut(),cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang * 180 / np.pi / 2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
# cv2.imshow('frame2',bgr)
# k = cv2.waitKey(30) & 0xff
# if k == 27:
# print "k == 27"
# break
# elif k == ord('s'):
cv2.imwrite('opticalfb.png',img_cut())
cv2.imwrite('opticalhsv.png',bgr)
prvs = next
