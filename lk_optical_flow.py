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


def lk_opflow():
    # ShiTomasi Corner Detection
    feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    # Params for LK optical flow
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    colour = np.random.randint(0,255,(100,3))

    for i in range(len(frames)):
        # Take first frame and find corners in it
        old_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)  # may need to be i-1
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        mask = np.zeros_like(frames[i - 1])

        frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), colour[j].tolist(), 2)
            frame = cv2.circle(frames[j],(a,b),5,colour[j].tolist(),-1)
            final_image = cv2.add(frame,mask)

        if i == 12003:
            cv2.imwrite('LK_.png',final_image)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)


img_cut()
lk_opflow()
