import cv2

img = cv2.imread('test_image.jpg')

def cannyFunc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


x = cannyFunc(img)

cv2.imshow('res',x)
cv2.waitKey(0)


# img = cv2.imread('test_image.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# blur  = cv2.GaussianBlur(gray, (5,5), 0)
# canny = cv2.Canny(blur, 50, 150)
# cv2.imshow('res',canny)
# cv2.waitKey(0)
