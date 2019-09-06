import numpy as np
import cv2
import os
import time

def checkposition(frame, imgname):
    # 原图转为hsv格式的灰度图
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 蓝色
    # lower_blue = np.array([100, 43, 46])
    # upper_blue = np.array([124, 255, 255])
    lower_blue = np.array([90, 43, 46])
    upper_blue = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 绘制出转换后的图像
    cv2.imwrite('./position/'+imgname+'.jpg', mask)
    # 固定阈值二值化处理
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    # 进行进行腐蚀操作
    # binary = cv2.erode(binary, None, iterations=2)
    # 进行膨胀操作
    binary = cv2.dilate(binary, None, iterations=2)
    # 检测人脸轮廓
    image, contours, hierarchv = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 获取原图宽高，这里是原图，不是方框图像
    width = mask.shape[1]
    height = mask.shape[0]
    # 获取二值化后的峰值坐标点（白色部分）
    minx = width
    miny = height
    maxx = 0
    maxy = 0

    # 遍历人脸轮廓
    for array1 in contours:
        # 计算人脸峰值坐标点位置
        for array2 in array1:
            for position in array2:
                if minx > position[0]:
                    if position[0] > 0:
                        minx = position[0]
                if miny > position[1]:
                    if position[1] > 0:
                        miny = position[1]

                if maxx < position[0]:
                    maxx = position[0]
                if maxy < position[1]:
                    maxy = position[1]


    # 计算原图中心点坐标
    centerx = width / 2
    centery = height / 2
    print('\033[0morigincenter:', centerx, centery)
    # 计算人脸中心坐标点
    maskcenterx = (maxx - minx) / 2 + minx
    maskcentery = (maxy - miny) / 2 + miny
    print('maskcenter:', maskcenterx, maskcentery)

    # 偏移量为宽高的1/5
    offsetx = width / 5
    offsety = height / 10
    # 根据偏移量，计算位置
    if maskcenterx - centerx > offsetx:
        if maskcentery - centery > offsety:
            position = 'right、down'
        elif centery - maskcentery > offsety:
            position = 'right、up'
        else:
            position = 'right'
    elif centerx - maskcenterx > offsetx:
        if maskcentery - centery > offsety:
            position = 'left、down'
        elif centery - maskcentery > offsety:
            position = 'left、up'
        else:
            position = 'left'
    else:
        position = 'center'
    print("position：\033[32m" + position)

    return position


# 各类数据模型地址https://github.com/opencv/opencv/tree/master/data/haarcascades
classfier = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
def calculate(frame, imgname):
    # position = checkposition(frame, imgname)

    # 通过open cv检测是否有人脸
    rectangles = classfier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
    # 遍历所有方框，计算所有方框面积总和
    totalarea = 0
    facewh = 0
    minfacex, minfacey, maxfacex, maxfacey = 0, 0, 0, 0
    for (x, y, w, h) in rectangles:
        # 方框框出人脸
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if facewh < w:
            facewh = w
            minfacex = x
            minfacey = y
            maxfacex = x + w
            maxfacey = y + h
    # 裁剪方框框出部分图片
    frame = frame[minfacey:maxfacey, minfacex:maxfacex]
    # 计算所有方框面积总和
    totalarea += (facewh * facewh)
    print('\033[0m\ntotalarea：', totalarea)
    # 绘制出带方框人脸的图片
    cv2.imwrite('./area/'+imgname+'_rect.jpg', frame)

    if len(rectangles) <= 0:
        print('\033[0m\nThe image has no face')
        return False
    else:
        # 原图转为hsv格式的灰度图
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 蓝色
        # lower_blue = np.array([100, 43, 46])
        # upper_blue = np.array([124, 255, 255])
        lower_blue = np.array([90, 43, 46])
        upper_blue = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 绘制出转换后的图像
        cv2.imwrite('./area/'+imgname+'.jpg', mask)
        # 固定阈值二值化处理
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # 进行进行腐蚀操作
        # binary = cv2.erode(binary, None, iterations=2)
        # 进行膨胀操作
        binary = cv2.dilate(binary, None, iterations=2)
        # 检测人脸轮廓
        # image, contours, hierarchv = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 所有白色面积总和
        facesarea = cv2.countNonZero(binary.copy())
        # 遍历人脸轮廓
        # for i in range(len(contours)):
            # 计算所有白色面积总和
            # facesarea += cv2.contourArea(contours[i])
        # contourArea计算的并不是轮廓的实际面积，可以使用countNonZero或者contour.size()来计算实际的像素面积

        # 计算所有人脸面积占比所有方框面积
        print('facesarea：', int(facesarea))
        # 计算比例，保留小数点后两位
        percent = round(facesarea / totalarea * 100, 2)
        print('percent:\033[32m' + str(percent) + '%\n')

        # print('\033[0m\nThe judging condition is the "percent > 65" and the "position == center"')
        # if percent > 65 and position == 'center':
        print('\033[0m\nThe judging condition is the "percent > 65"')
        if percent > 65:
            return True
        else:
            return False


def main():
    filesdir = './src/'
    list = os.listdir(filesdir)
    for i in range(0, len(list)):
        imgpath = os.path.join(filesdir, list[i])
        imgname = os.path.basename(list[i])
        if imgname.endswith('uv'):
            print(imgname)
            if os.path.exists(imgpath):
                frame = cv2.imread(imgpath)
                print('result: \033[34m', calculate(frame, imgname))
            else:
                print('\033[31mImage not exist')

            print('\033[0m-------------------------------------------------------')
            time.sleep(0.1)

    # while True:
    #     print('\nPlease input image name，no suffix')
    #     imgname = input('imagename: ')
    #     filename = './src/' + imgname
    #     if os.path.exists(filename):
    #         frame = cv2.imread(filename)
    #         print('result:\033[34m', calculate(frame, imgname))
    #     else:
    #         print('\033[31mImage not exist')
    #
    #     print('\033[0m-------------------------------------------------------')


if __name__ == '__main__':
    main()


