import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test_image.jpg')
img_cpy = np.copy(img)

def make_coordinates(image, line_parameters):
    slope ,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis= 0)  #return [slope, intercept]
    right_fit_average = np.average(right_fit, axis= 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def cannyFunc(img):
    gray = cv2.cvtColor(img_cpy, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):   # return lines
    line_image = np.zeros_like(img)  # عکس را کامل مشکی میکند
    if lines is not None:
        for line in lines:
            #print(line)  #2D
            #line = line.reshape(4) #1D #یعنی چهار ستونی که الان هست رو بکن 4 المان
            x1, y1, x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) # خطوط را به رنگ آبی با قطر 10 میکشد
    return line_image

def region_of_interst(img):
    heigh = img.shape[0]  #بدست آوردن ارتفاع
    triangle = np.array([ [(200,heigh), (1100, heigh), (550, 250)] ]) # رسم مثلث
    mask = np.zeros_like(img) # سیاه کردن تصویر
    cv2.fillPoly(mask, triangle, 255) #با هم  یکی میکند ؟؟؟؟
    mask_img= cv2.bitwise_and(img, mask)
    return mask_img



# canny = cannyFunc(img_cpy)
# cropped_img= region_of_interst(canny)
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # کشیدن بهرین خط
# line_image = display_lines(img_cpy, lines)
# cv2.imshow('result', line_image)
# cv2.waitKey(0)


canny = cannyFunc(img_cpy)
cropped_img= region_of_interst(canny)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # کشیدن بهرین خط
averaged_lines = average_slope_intercept(img_cpy, lines)
line_image = display_lines(img_cpy, averaged_lines)
combo_image = cv2.addWeighted(img_cpy,0.8,  line_image,1  , 1)  # تو تصویر را ترکیب میکند با درصد محو شدگی هر کدام
cv2.imshow('result', combo_image)  # نمایش دادن تصاویر ترکیب شده
cv2.waitKey(0)
