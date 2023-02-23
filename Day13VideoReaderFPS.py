"""
This Code is written for Vjdeo reading

import numpy as np
import cv2

print("Package Imported")

# cap = cv2.VideoCapture("C:/Users/user/Desktop/Computer_Vision_Proj/Task_Code.mp4")
# cap = cv2.VideoCapture("C:/Users/user/Desktop/Computer_Vision_Proj/cars.avi")
# cap = cv2.VideoCapture("C:/Users/user/Desktop/Computer_Vision_Proj/Icl2.mp4")
cap = cv2.VideoCapture("BagWeightVid.mp4")
FPS = 0
cap.set(cv2.CAP_PROP_FPS, FPS)


count = 0

while True:
    success, img = cap.read()
    img = img[450:530, 435:568]

    # cv2.rectangle(img, (0, 0), (850, 700), (0, 0, 255), 4)
    # img = cv2.resize(img, (950, 650))
    cv2.imwrite("./Captured/frame%d.jpg" % count, img)

    cv2.imshow("Video", img)
    # if cv2.waitKey(10) == 27:
    #     break
    # Here this q is for quit the video.
    if cv2.waitKey(600) & 0xFF == ord('q'):
        break
    count += 1

"""
import datetime
import numpy as np
import cv2

vidcap = cv2.VideoCapture('BagWeightVid.mp4')


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    # image = image[450:530, 435:568]
    image = image[450:530, 415:568]
    cv2.imshow("Video", image)
    if hasFrames:
        cv2.imwrite("./Captured/newframe" + str(sec) + "_sec.jpg", image)  # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 0.95  # it will capture image in each 0.5 second
success = getFrame(sec)
while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
