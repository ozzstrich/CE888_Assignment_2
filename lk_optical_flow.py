import cv2
import numpy as np

frames = []


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
            frames.append(frame)
    return frame

img_cut()
print frames[400]
