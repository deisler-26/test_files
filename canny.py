import numpy as np
import cv2

img_BGR = cv2.imread("hoge.jpg")

img_Lab = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2Lab)
img_Lab_L, img_Lab_a, img_Lab_b = cv2.split(img_Lab)
# cv2.imwrite('hoge.a.jpg', img_Lab_L)

img_canny = cv2.Canny(img_Lab_a, 30, 35)
cv2.imwrite('hoge_canny.jpg', img_canny)
