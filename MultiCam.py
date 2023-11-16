import cv2
import numpy as np
import imutils
from datetime import datetime
import os
import shutil

# These following parameters is used to control the behavior of the system
# If you wish to run the program in default settings, no change needs to be made
NUM_CAMERAS = 1  # Number of cameras connected to the system
THRESHOLD_MOTION = 0.1  # Threshold for Passingby target
THRESHOLD_APPROACH = 3  # Threshold for Approaching target
TIMEOUT = 100000  # timeout to reinit the system
DRAW_TRAINGLE = False
IMSHOW = True  # Whether to show on monitor

# Find frame difference basing on GMM and return the ratio of contour change
def ROI_detect(frame_1, fgbg, CameraID):
    # Detect by GMM
    fgmask = fgbg.apply(frame_1)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    thresh2 = cv2.threshold(fgmask, 16, 255, cv2.THRESH_BINARY)[1]


    # Edge Detection
    # change thresh/thresh to apply different edge detect algorithm
    cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Init boundry
    (minX, minY) = (10000, 10000)
    (maxX, maxY) = (-10000, -10000)

    # loop over the contours
    ROI_detected = False
    contour = 0  # set this param to 1 to avoid dividing zero

    for c in cnts:
        contour = max(contour, cv2.contourArea(c))
        # compute the bounding box of the contour
        if DRAW_TRAINGLE is True:
            (x, y, w, h) = cv2.boundingRect(c)
            # reduce noise by enforcing requirements on the bounding box size
            if w > 30 and h > 30:
                # update our bookkeeping variables
                ROI_detected = True
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x + w - 1)
                maxY = max(maxY, y + h - 1)

    # draw a rectangle surrounding the region of motion
    if DRAW_TRAINGLE is True:
        cv2.rectangle(frame_1, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
    return frame_1, contour

# Detect whether there is a target passingby
def Motion_detect(contour, cnt_area, CameraID, motion_alert_rev, resolution, MOTION_TRIGGER, GLOBAL_CNT, GLOBAL_RESET,
                  APPROACH, frame, save_cnt, global_det_cnt, TOWARN):
    # Global timeout
    if GLOBAL_CNT == 0:
        GLOBAL_RESET = True
    if GLOBAL_RESET is True:
        GLOBAL_CNT = GLOBAL_CNT + 1
        if GLOBAL_CNT >= TIMEOUT:
            GLOBAL_RESET = False
        return 0, False, False, GLOBAL_CNT, GLOBAL_RESET

    # detect passingby target
    GLOBAL_CNT = GLOBAL_CNT - 1
    if GLOBAL_CNT > 0 and GLOBAL_RESET is False:
        # Check if object is a human face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 100 * contour / resolution > THRESHOLD_MOTION:
            cnt_area[CameraID] = min(15, cnt_area[CameraID] + 2)
            if cnt_area[CameraID] >= 4:  #
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    MOTION_TRIGGER = True
                    if motion_alert_rev is False:
                        print("TARGET PASSINGBY!")
                        save_cnt = 10
                        motion_alert_rev = True
        cnt_iter = True

        for i in range(NUM_CAMERAS):
            if cnt_area[i] > 2:
                cnt_iter = False

        if cnt_iter is True:
            MOTION_TRIGGER = False
            GLOBAL_CNT = TIMEOUT
            if motion_alert_rev is True:
                APPROACH = True
                if TOWARN:
                    old_path_name = R"./captured_image/" + datetime.now().strftime("%F/") + str(global_det_cnt)
                    shutil.move(old_path_name, old_path_name + "-approach")
                    TOWARN = False
                save_cnt = 0
                print("ALL CLEAR!")
                motion_alert_rev = False
        cnt_area[CameraID] = max(0, cnt_area[CameraID] - 1)
    return cnt_area[CameraID], motion_alert_rev, MOTION_TRIGGER, GLOBAL_CNT, GLOBAL_RESET, APPROACH, save_cnt, TOWARN

# Detect whether there is a target approaching
def Approach(contour, sum_block, cnt_block, queue, resolution, APPROACH, TOWARN):
    buffer = 100 * contour / resolution
    if buffer < THRESHOLD_APPROACH:
        approach_flag = False
    else:
        approach_flag = True
    if approach_flag is True:
        if APPROACH is True:
            print("TARGET APPROACHING!")
            TOWARN = True
            APPROACH = False
    return sum_block, cnt_block, queue, APPROACH, TOWARN

# Save image
def save_img(save_cnt, image, number_of_detect, path=R"./captured_image"):
    if save_cnt > 0:
        if save_cnt == 10:
            number_of_detect += 1
        img_path = path + R"/" + datetime.now().strftime("%F/") + str(number_of_detect)
        if not os.path.exists(img_path + "_approach"):
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            file_num = sum([os.path.isfile(os.path.join(img_path, file)) for file in os.listdir(img_path)])
            cv2.imwrite(img_path + R"/" + str(file_num + 1) + ".jpg", image)

            save_cnt -= 1
    return save_cnt, number_of_detect

# Real time video surveillance algorithm
def video(CameraID):
    cap = []
    fgbg = []
    resolution = []

    # Param init for motion detection
    cnt_area = []
    motion_alert_rev = False
    MOTION_TRIGGER = False

    # Param for Approach detection
    sum_block = []
    cnt_block = []
    queue = []

    # RESULT
    result = []
    area = []
    contour = []
    save_cnt = []

    # Save_img
    global_det_cnt = 0

    if NUM_CAMERAS == 1:
        CAMERA_MEMBER_LIST = [0]
    elif NUM_CAMERAS == 2:
        CAMERA_MEMBER_LIST = [0, 2]
    elif NUM_CAMERAS == 3:
        CAMERA_MEMBER_LIST = [0, 2, 4]
    elif NUM_CAMERAS == 4:
        CAMERA_MEMBER_LIST = [0, 2, 4, 6]

    # Data sturct init
    for i in range(NUM_CAMERAS):
        cap.append(cv2.VideoCapture(CAMERA_MEMBER_LIST[i]))
        fgbg.append(cv2.createBackgroundSubtractorMOG2())
        ret, frame = cap[i].read()
        L, W, H = frame.shape
        resolution.append(L * W)

        cnt_area.append(0)
        sum_block.append(0)
        cnt_block.append(0)
        queue.append([])

        result.append(0)
        area.append(0)
        contour.append(0)
        save_cnt.append(0)

    keep_open = True

    # TIMEOUT
    global_count = TIMEOUT
    reset = False
    APPROACH = True
    TOWARN = False

    while cap[0].isOpened() and keep_open:
        ret, frame = cap[CameraID].read()
        result[CameraID], contour[CameraID] = ROI_detect(frame, fgbg[CameraID], CameraID)
        if IMSHOW:
            cv2.imshow(str(CameraID), frame)
        exit_key = cv2.waitKey(1)

        # Motion Detection
        cnt_area[CameraID], motion_alert_rev, MOTION_TRIGGER, global_count, reset, APPROACH, save_cnt[
            CameraID], TOWARN = \
            Motion_detect(contour[CameraID], cnt_area, CameraID, motion_alert_rev, resolution[CameraID], MOTION_TRIGGER,
                          global_count,
                          reset, APPROACH, frame, save_cnt[CameraID], global_det_cnt, TOWARN)

        # Approaching Detection
        if MOTION_TRIGGER:
            sum_block[CameraID], cnt_block[CameraID], queue[CameraID], APPROACH, TOWARN = \
                Approach(contour[CameraID], sum_block[CameraID], cnt_block[CameraID], queue[CameraID],
                         resolution[CameraID], APPROACH, TOWARN)
        else:
            sum_block[CameraID] = 0
            cnt_block[CameraID] = 0
            queue[CameraID] = []

        save_cnt[CameraID], global_det_cnt = save_img(save_cnt[CameraID], frame, global_det_cnt)
        CameraID = (CameraID + 1) % NUM_CAMERAS
        if exit_key == 27:
            keep_open = False
            cv2.destroyAllWindows()
    for i in range(NUM_CAMERAS):
        cap[i].release()


if __name__ == '__main__':
    # Load Haar cascade classifier for human faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Start")
    video(0)  # init from the first camera
