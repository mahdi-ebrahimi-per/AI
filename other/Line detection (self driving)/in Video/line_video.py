import cv2
import numpy as np
import matplotlib.pyplot as plt

#این فانکشن برای این است که تصویرمان را در بهترین حالت ترشولد کنیم
def cannyFunc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 1) سیاه سفید میکنیم
    blur = cv2.GaussianBlur(gray, (5,5), 0) # 2) تار میکنیم تا نویز ها گرفته شود
    canny = cv2.Canny(blur, 50, 150) # 3) ترشولد
    return canny




def display_lines(image, lines):   # return lines
    #خط را میگیرد و در نهایت عکس سیاه همراه با خط کشیده تحویل میدهد
    line_image = np.zeros_like(image)  # عکس را کامل مشکی میکند
    if lines is not None: #
        for line in lines:
            #print(line)  #2D
            #line = line.reshape(4) #1D #یعنی چهار ستونی که الان هست رو بکن 4 المان
            x1, y1, x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) # خطوط را به رنگ آبی با قطر 10 میکشد
    return line_image



#این فانکشن برای این است که فقط جلوی ماشین را بگیریم و کوه و بقیه چیز ها در تصویر نباشد
def region_of_interst(img):
    heigh = img.shape[0]  #بدست آوردن ارتفاع
    triangle = np.array([ [(200,heigh), (1100, heigh), (550, 250)] ]) # رسم مثلث
    mask = np.zeros_like(img) # سیاه کردن تصویر
    cv2.fillPoly(mask, triangle, 255) #با هم  یکی میکند ؟؟؟؟
    mask_img= cv2.bitwise_and(img, mask)
    return mask_img


cap = cv2.VideoCapture('test2.mp4')   # video
#ap = cv2.VideoCapture(0)               #camera

while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = cannyFunc(frame)
    cropped_img= region_of_interst(canny_image)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # کشیدن بهرین خط
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame,0.8,  line_image,1  , 1)  # تو تصویر را ترکیب میکند با درصد محو شدگی هر کدام
    cv2.imshow('result', combo_image)  # نمایش دادن تصاویر ترکیب شده
    cv2.waitKey(1) #برای اینکه فیلم فریم هاش عوض شه
    if cv2.waitKey(1) == ord('q'): # برای خروج از فیلم
        break
#cap.relese()
cv2.destroyAllWindows()




canny = cannyFunc(img_cpy)
cropped_img= region_of_interst(canny_image)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # کشیدن بهرین خط
line_image = display_lines(img_cpy, lines)
print(line_image)
cv2.imshow('result', line_image)
cv2.waitKey(0)
