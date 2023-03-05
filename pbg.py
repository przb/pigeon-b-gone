import cv2
import numpy as np

# array = np.full((400, 600), 0.3)

pic = cv2.imread("gorilla.jpeg")
blur_kernel = np.full((3, 3), 1 / 9)
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
cnn_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

pic = cv2.filter2D(pic, -1, gaussian_kernel)
pic = cv2.filter2D(pic, -1, gaussian_kernel)
cv2.imshow("Win2", pic)

pic = cv2.filter2D(pic, -1, cnn_filter)
cv2.imshow("Win", pic)

cv2.waitKey()
cv2.destroyAllWindows()
