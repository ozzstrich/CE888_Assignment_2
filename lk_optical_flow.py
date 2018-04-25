import cv2
import numpy as np

frames = []


# Read in images and cut into 4 slices
def img_cut():

    for i in range(3001):
        image_name = 'Data/Breakout/breakout' + str(i) + '.png'
        img = cv2.imread(image_name)

        # set params for crop
        crop_height = 84
        crop_y = 0
        crop_width = 84
        crop_x = 0

        for i in range(4):
            frame = img[crop_y:crop_height, crop_x:crop_width]
            crop_x += crop_height
            crop_width += crop_height
            frames.append(frame)
    return frame


def lk_opflow():
    # ShiTomasi Corner Detection parameters
    feature_params = dict(maxCorners = 10, qualityLevel = 0.5, minDistance = 0.5, blockSize = 2)

    # LK optical flow parameters
    lk_params = dict(winSize = (10,10), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    colour = np.random.randint(0,255,(100,3))

    for i in range(len(frames)):

        # get good points frmo previous img
        previous_g_img = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        point0 = cv2.goodFeaturesToTrack(previous_g_img, mask = None, **feature_params)

        # Create mask array of same shape
        mask = np.zeros_like(frames[i - 1])

        current_g_img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Optical flow calculation using the previous and current image, given the point and the params
        point1, st, err = cv2.calcOpticalFlowPyrLK(previous_g_img, current_g_img, point0, None, **lk_params)

        # Select good points
        current_good_point = point1[st == 1]
        old_good_point = point0[st == 1]

        for j, (new, old) in enumerate(zip(current_good_point, old_good_point)):
            # get points
            a,b = new.ravel()
            c,d = old.ravel()

            # apply lines using points and colour
            mask = cv2.line(mask, (a,b),(c,d), colour[j].tolist(), 2)
            frame = cv2.circle(frames[j],(a,b),2,colour[j].tolist(),-1)

            final_image = cv2.add(frame,mask)

        if i == len(frames) - 1:
            cv2.imwrite('LK_.png',final_image)

            previous_g_img = current_g_img.copy()
            point0 = current_good_point.reshape(-1,1,2)


img_cut()
lk_opflow()
