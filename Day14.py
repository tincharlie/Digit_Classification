from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import PIL
# from PIL import Image as im
from PIL import ImageEnhance, Image

import numpy as np
import matplotlib.pyplot as plt
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITSDICT = {
    (1, 1, 0, 1, 1, 0, 0): 0,
    (1, 1, 1, 0, 1, 1, 1): 0,
    (1, 1, 0, 1, 1, 1, 0): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 1,
    (1, 0, 0, 1, 1, 1, 0): 2,

    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
    (1, 1, 1, 1, 0, 1, 0): 0,
    (0, 1, 1, 1, 1, 0, 0): 5
}

# image = cv2.imread("C:/Users/User/Downloads/w1.jpeg")
# image = cv2.imread("C:/Users/user/Desktop/Computer_Vision_Proj/Captured/newframe1.9_sec.jpg")
image = cv2.imread("C:/Users/user/Desktop/Computer_Vision_Proj/Captured/newframe9.5_sec.jpg")

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
# image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(blurred, 50, 200, 255)
edged = cv2.Canny(blurred, 50, 200, 255)

data233 = Image.fromarray(gray)
# data233 = Image.fromarray(image)
im = data233

# Setting the points for cropped image
# left = 395
# top = 150
# right = 610
# bottom = 300
left = 0
top = 0
right = 0
bottom = 0

# Cropped image of above dimension
# (It will not change original image)
# im1 = im.crop((left, top, right, bottom))
# im1 = im[450:530, 435:568]
# im_data = np.asarray(im)
im_data = np.asarray(data233)

roi_color = im_data
# roi_color = cv2.imread("inter/download.png")
# roi_color = cv2.imread("inter/ocbc-roi.png")
# roi_color = cv2.imread("inter/1.png")
# roi_color = cv2.imread("inter/download.png")
# img = Image.open("inter/1.png")
# img.show()
# filter = ImageEnhance.Brightness(img)
#
# im1 = img.filter(1.2)

# create a sharpening kernel
# sharpen_filter = np.array([[-1, -1, -1],
#                            [-1, 9, -1],
#                            [-1, -1, -1]])
# # applying kernels to the input image to get the sharpened image
# roi_color = cv2.filter2D(roi_color, -1, sharpen_filter)
#cv2.imshow('Required image', roi_color)

# roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
roi = cv2.cvtColor(np.asarray(data233), cv2.COLOR_BGR2GRAY)
RATIO = roi.shape[0] * 0.1

roi = cv2.bilateralFilter(roi, 5, 30, 60)

trimmed = roi[int(RATIO):, int(RATIO): roi.shape[1] - int(RATIO)]
roi_color = roi_color[int(RATIO):, int(RATIO): roi.shape[1] - int(RATIO)]

edged = cv2.adaptiveThreshold(
    trimmed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
dilated = cv2.dilate(edged, kernel, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
dilated = cv2.dilate(dilated, kernel, iterations=1)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1), )
eroded = cv2.erode(dilated, kernel, iterations=1)


h = roi.shape[0]
ratio = int(h * 0.07)
eroded[-ratio:, ] = 0
eroded[:, :ratio] = 0



cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digits_cnts = []

canvas = trimmed.copy()
cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)


canvas = trimmed.copy()
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if h > 30:
        digits_cnts += [cnt]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.drawContours(canvas, cnt, 0, (255, 255, 255), 1)


print(f"No. of Digit Contours: {len(digits_cnts)}")


sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

canvas = trimmed.copy()

for i, cnt in enumerate(sorted_digits):
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
    cv2.putText(canvas, str(i), (x, y - 3), FONT, 0.3, (0, 0, 0), 1)

digits = []
canvas = roi_color.copy()
for cnt in sorted_digits:
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = eroded[y: y + h, x: x + w]
    print(f"W:{w}, H:{h}")
    # convenience units
    qW, qH = int(w * 0.25), int(h * 0.15)
    fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)

    # seven segments in the order of wikipedia's illustration
    sevensegs = [
        ((0, 0), (w, qH)),  # a (top bar)
        ((w - qW, 0), (w, halfH)),  # b (upper right)
        ((w - qW, halfH), (w, h)),  # c (lower right)
        ((0, h - qH), (w, h)),  # d (lower bar)
        ((0, halfH), (qW, h)),  # e (lower left)
        ((0, 0), (qW, halfH)),  # f (upper left)
        # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
        (
            (0 + fractionW, halfH - fractionH),
            (w - fractionW, halfH + fractionH),
        ),  # center
    ]

    # initialize to off
    on = [0] * 7

    for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
        region = roi[p1y:p2y, p1x:p2x]
        print(
            f"{i}: Sum of 1: {np.sum(region == 255)}, Sum of 0: {np.sum(region == 0)}, Shape: {region.shape}, Size: {region.size}"
        )
        if np.sum(region == 255) > region.size * 0.5:
            on[i] = 1
        print(f"State of ON: {on}")

    digit = DIGITSDICT[tuple(on)]
    print(f"Digit is: {digit}")
    digits += [digit]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CYAN, 1)
    cv2.putText(canvas, str(digit), (x - 5, y + 6), FONT, 0.3, (0, 0, 0), 1)
    cv2.imshow("Digit", canvas)
    cv2.waitKey(0)


# cv2.destroyAllWindows()
print(f"Digits on the token are: {digits}")
print(u"{}{}.{}{} \u00b0kg".format(*digits))
