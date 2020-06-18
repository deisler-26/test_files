import numpy as np
import cv2

img_BGR = cv2.imread("hoge.jpg")

# L*a*b*
img_Lab = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2Lab)
img_Lab_L, img_Lab_a, img_Lab_b = cv2.split(img_Lab)

# use a* channel to get edge (Sobel)
img_sobel_x = cv2.Sobel(img_Lab_a, cv2.CV_8U, 1, 0, ksize=5)
img_sobel_y = cv2.Sobel(img_Lab_a, cv2.CV_8U, 0, 1, ksize=5)
img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 1)
cv2.imwrite('hoge_sobel.jpg', img_sobel)
