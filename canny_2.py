import numpy as np
import cv2

img_BGR = cv2.imread("hoge.jpg")

# L*a*b*
img_Lab = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2Lab)
img_Lab_L, img_Lab_a, img_Lab_b = cv2.split(img_Lab)

# detect green area
_thres, img_green = cv2.threshold(img_Lab_a, 110, 255, cv2.THRESH_BINARY_INV)

# remove small area
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_green, connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1
img_plant = np.zeros((img_green.shape))
for i in range(0, nb_components):
    if sizes[i] >= 3000:
        img_plant[output == i + 1] = 255
img_plant = np.uint8(img_plant)

img_plant_canny = cv2.Canny(img_plant, 0, 255)
cv2.imwrite('hoge_canny_2.jpg', img_plant_canny)
