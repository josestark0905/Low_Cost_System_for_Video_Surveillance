import cv2
import numpy as np
import imutils
from queue import Queue
from datetime import datetime
import os


def save_img(image, number_of_detect, path=R"./captured_image"):
    img_path = path + R"/" + datetime.now().strftime("%F/") + str(number_of_detect)
    # img_path = path
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    file_num = sum([os.path.isfile(os.path.join(img_path, file)) for file in os.listdir(img_path)])
    # print(file_num)
    cv2.imwrite(img_path + R"/" + str(file_num + 1) + ".jpg", image)


def save_video(number_of_frame, path_image=R"./captured_image", path_video=R"./captured_video"):
    limit_name = str(number_of_frame) + ".jpg"
    if limit_name in os.listdir(path_image):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if not os.path.exists(path_video):
            os.makedirs(path_video)
        path_video += R"/output.avi"
        out = cv2.VideoWriter(path_video, fourcc, 200.0, (640, 480))
        for i in range(1, 10):
            image = path_image + R"/" + str(i) + ".jpg"
            if str(i) + ".jpg" in os.listdir(path_image):
                print(image)
                added_image = cv2.resize(cv2.imread(image), (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("added", added_image)
                cv2.waitKey(200)
                out.write(added_image)
        out.release()


def difference(image1, image2):
    dif = np.array(image1, dtype=np.int16)
    dif = np.abs(dif - image2)
    dif = np.array(dif, dtype=np.uint8)  # get different
    gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 56, 255, 0)[1]
    thresh = cv2.blur(thresh, (45, 45))
    if np.max(thresh) != 0:
        return True
    else:
        return False


def update_bg(bg, current_frame, frame_queue):
    if frame_queue.full():
        frame_queue.get()
    frame_queue.put(current_frame.copy())
    if not difference(bg.copy(), current_frame.copy()):
        result = current_frame.copy()
    else:
        if not frame_queue.empty() and not difference(frame_queue.get(), current_frame.copy()):
            result = current_frame.copy()
        else:
            result = bg.copy()
    # cv2.imshow("result", result)
    return result, frame_queue


def ROI_detect(frame_1, frame_2):
    # load the background and foreground images
    # convert the background and foreground images to grayscale
    # 灰度化处理
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("gray", 0)
    # cv2.imshow("gray", gray_1)

    # perform background subtraction by subtracting the foreground from
    # the background and then taking the absolute value
    # 背景减法
    sub = gray_1.astype("int32") - gray_2.astype("int32")
    sub = np.absolute(sub).astype("uint8")
    # cv2.namedWindow("sub", 0)
    # cv2.imshow("sub", sub)

    # threshold the image to find regions of the subtracted image with
    # larger pixel differences
    thresh = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations to remove noise
    # erode ,dilate 降噪处理
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=1)
    # cv2.namedWindow("threshold", 0)
    # cv2.imshow("threshold", thresh)
    # find contours in the thresholded difference map and then initialize
    # 发现边界
    # our bounding box regions that contains the *entire* region of motion
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 给边界初始值
    (minX, minY) = (10000000, 10000000)
    (maxX, maxY) = (-10000000, -10000000)
    if_draw = False
    # loop over the contours
    # 循环计算边界
    area = 0
    for c in cnts:
        # 轮廓面积
        area = max(area, cv2.contourArea(c))
        if_draw = True
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # reduce noise by enforcing requirements on the bounding box size
        # 如果边界值，w 或 h 小于20 就认为是噪音
        if w > 50 and h > 50:
            # update our bookkeeping variables
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x + w - 1)
            maxY = max(maxY, y + h - 1)

    # draw a rectangle surrounding the region of motion
    print(area)
    result = frame_2.copy()
    # 绘制长方形
    if if_draw:
        cv2.rectangle(result, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
    return result


def video(camera_id):
    cv2.namedWindow("video_" + str(camera_id), 0)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW only works for windows version
    keep_open = True
    last_ret, last_frame = cap.read()
    fq = Queue(maxsize=40)
    while cap.isOpened() and keep_open:
        ret, frame = cap.read()
        result = ROI_detect(last_frame, frame)
        save_img(result, 1)
        # save_video(10)
        cv2.imshow("video_" + str(camera_id), result)
        last_frame, fq = update_bg(last_frame, frame, fq)
        exit_key = cv2.waitKey(30)
        if exit_key == 27:
            keep_open = False
            cv2.destroyAllWindows()
    cap.release()
    # save_video(10)


if __name__ == '__main__':
    # print(datetime.now().strftime("%F-%H"))
    video(0)
