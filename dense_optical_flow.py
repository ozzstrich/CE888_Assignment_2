import cv2
import numpy as np

frames = []
uv = []

# TODO(Next Step) - Optical flow for all games, and derive conclusion. Maybe pass these images to LIME?

# Name of folder with game images and name of image (gamename)
folder = "James_Bond"
gamename = "jamesbond"


# Read in images and cut into 4 slices (mod so it does it for other games)
def img_cut():
    for i in range(3001):
        image_name = 'Data/' + folder + '/' + gamename + str(i) + '.png'
        img = cv2.imread(image_name)

        if i == 3000:
            print image_name

        crop_height = 84
        crop_y = 0
        crop_width = 84
        crop_x = 0

        for j in range(4):
            frame = img[crop_y:crop_height, crop_x:crop_width]
            # print crop_x, " : ", crop_width
            crop_x += crop_height
            crop_width += crop_height
            frames.append(frame)
    return frame


def background_sub():
    bgs = cv2.createBackgroundSubtractorMOG2()

    for i in range(len(frames)):
        frame = bgs.apply(frames[i])
        cv2.imwrite('Data/Background_Subtraction/BGS_' + gamename + ' ' + str(i) + ' .png', frame)



def dense_of():
    for i in range(len(frames)):  # change range to (len(frames when working))
        print i, "/", len(frames)
        prvs = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frames[i])
        hsv[...,1] = 255
        next = cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        uv.append(flow[0])

        if i == len(frames) - 1:
            print "Writing image"
            cv2.imwrite('DO_' + gamename + '.png',bgr)
            print "Image done"
            print flow
            print len()
        prvs = next


img_cut()
background_sub()
# dense_of()
