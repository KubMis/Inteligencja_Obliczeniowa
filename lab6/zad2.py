# pip install opencv-python
import cv2
import numpy as np

if __name__ == "__main__":
    image = cv2.imread('birds.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    R, G, B = cv2.split(image)
    gray_image2 = np.round((R + G + B) / 3).astype(np.uint8)

    cv2.imwrite('birds_gray.png', gray_image)
    cv2.imwrite('birds_gray2.png', gray_image2)