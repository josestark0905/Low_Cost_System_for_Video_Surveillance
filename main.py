import cv2
import numpy as np
import imutils
from queue import Queue


def update_bg(bg, current_frame, frame_queue):
    if campHash(dhash(bg), dhash(current_frame)) < 6:
        frame_queue.put(bg.copy())
        bg = current_frame
    else:
        if not frame_queue.empty():
            if campHash(dhash(frame_queue.get()), dhash(current_frame)) < 6:
                frame_queue.put(bg.copy())
                bg = current_frame
    return bg, frame_queue


# 差异值哈希算法
def dhash(image):
    # 将图片转化为8*8
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result


# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def background(bg, fg):
    # load the background and foreground images
    # convert the background and foreground images to grayscale
    # 灰度化处理
    bgGray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    fgGray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

    # perform background subtraction by subtracting the foreground from
    # the background and then taking the absolute value
    # 背景减法
    sub = bgGray.astype("int32") - fgGray.astype("int32")
    sub = np.absolute(sub).astype("uint8")

    # threshold the image to find regions of the subtracted image with
    # larger pixel differences
    thresh = cv2.threshold(sub, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations to remove noise
    # erode ,dilate 降噪处理
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=1)

    # find contours in the thresholded difference map and then initialize
    # 发现边界
    # our bounding box regions that contains the *entire* region of motion
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 给边界初始值
    (minX, minY) = (10000, 10000)
    (maxX, maxY) = (-10000, -10000)

    # loop over the contours
    # 循环计算边界
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # reduce noise by enforcing requirements on the bounding box size
        # 如果边界值，w 或 w 小于20 就认为是噪音
        if w > 20 and h > 20:
            # update our bookkeeping variables
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x + w - 1)
            maxY = max(maxY, y + h - 1)

    # draw a rectangle surrounding the region of motion
    # 绘制长方形
    cv2.rectangle(fg, (int(minX), int(minY)), (int(maxX), int(maxY)), (0, 255, 0), 2)
    return fg


def Gaussian_adaptive(img_original):
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    # img_blur = cv2.GaussianBlur(img_original, (5, 5), 20)
    img_blur = cv2.bilateralFilter(img_original, 7, 25, 50)
    # 自适应阈值分割
    img_thresh = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    img_thresh_blur = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    return img_thresh_blur


def grab_edge_1(img):
    # 读取文件
    mat_img = img
    mat_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应分割
    dst = cv2.adaptiveThreshold(mat_img2, 210, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY_INV, 3, 10)
    # 提取轮廓
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 标记轮廓
    cv2.drawContours(mat_img, contours, -1, (255, 0, 255), 3)
    return mat_img


def grab_edge(img):
    # 边缘提取
    edge = cv2.Canny(np.uint8(img), 50, 100, apertureSize=3, L2gradient=True)
    # 提取轮廓
    '''
    findcontour()函数中有三个参数，第一个img是源图像，第二个model是轮廓检索模式，第三个method是轮廓逼近方法。输出等高线contours和层次结构hierarchy。
    model:  cv2.RETR_EXTERNAL  仅检索极端的外部轮廓。 为所有轮廓设置了层次hierarchy[i][2] = hierarchy[i][3]=-1
            cv2.RETR_LIST  在不建立任何层次关系的情况下检索所有轮廓。
            cv2.RETR_CCOMP  检索所有轮廓并将其组织为两级层次结构。在顶层，组件具有外部边界；在第二层，有孔的边界。如果所连接零部件的孔内还有其他轮廓，则该轮廓仍将放置在顶层。
            cv2.RETR_TREE  检索所有轮廓，并重建嵌套轮廓的完整层次。
            cv2.RETR_FLOODFILL  输入图像也可以是32位的整型图像(CV_32SC1)
    method：cv2.CHAIN_APPROX_NONE  存储所有的轮廓点，任何一个包含一两个点的子序列（不改变顺序索引的连续的）相邻。
            cv2.CHAIN_APPROX_SIMPLE  压缩水平，垂直和对角线段，仅保留其端点。 例如，一个直立的矩形轮廓编码有4个点。
            cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS 近似算法
    '''
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓 第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    dst = np.ones(img.shape, dtype=np.uint8)
    cv2.drawContours(dst, contours, -1, (0, 255, 0), 1)
    # 绘制单个轮廓
    '''cnt = contours[50]
    cv2.drawContours(dst, [cnt], 0, (0, 0, 255), 1)'''
    '''
    # 特征矩
    cnt = contours[50]
    M = cv2.moments(cnt)
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)  # 绘制圆点

    # 轮廓面积
    area = cv2.contourArea(cnt)
    print(area)

    # 轮廓周长：第二个参数指定形状是闭合轮廓(True)还是曲线
    perimeter = cv2.arcLength(cnt, True)
    print(perimeter)

    # 轮廓近似：epsilon是从轮廓到近似轮廓的最大距离--精度参数
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.polylines(dst, [approx], True, (0, 255, 255))  # 绘制多边形
    print(approx)

    # 轮廓凸包：returnPoints：默认情况下为True。然后返回凸包的坐标。如果为False，则返回与凸包点相对应的轮廓点的索引。
    hull = cv2.convexHull(cnt, returnPoints=True)
    cv2.polylines(dst, [hull], True, (255, 255, 255), 2)  # 绘制多边形
    print(hull)

    # 检查凸度：检查曲线是否凸出的功能，返回True还是False。
    k = cv2.isContourConvex(cnt)
    print(k)

    # 边界矩形:最小外接矩形
    # 直角矩形
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # 旋转矩形
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)

    # 最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(dst, center, radius, (0, 255, 0), 2)

    # 拟合椭圆
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(dst, ellipse, (0, 0, 255), 2)

    # 拟合直线
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(dst, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)'''
    return dst


if __name__ == '__main__':
    frame_queue = Queue()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    keep_open = True
    bg_ret, bg_frame = cap.read()
    while cap.isOpened() and keep_open:
        ret, frame = cap.read()
        # Gframe = Gaussian_adaptive(frame)
        # Glast_frame = Gaussian_adaptive(last_frame)
        # result = grab_edge(frame - last_frame)
        # result = grab_edge(Gframe, Glast_frame)
        # result=grab_edge_1(frame)
        result = background(frame, bg_frame)
        bg_frame, frame_queue = update_bg(bg_frame, frame, frame_queue)
        cv2.imshow("video1", result)
        exit_key = cv2.waitKey(30)
        if exit_key == 27:
            keep_open = False
            cv2.destroyAllWindows()
    cap.release()
