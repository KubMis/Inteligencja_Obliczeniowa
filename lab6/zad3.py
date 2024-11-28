import os
import cv2
from matplotlib import pyplot as plt

path_to_image = "bird_miniatures_unzipped"
images_directory = os.listdir(path_to_image)

for image_name in images_directory:
    image = cv2.imread(os.path.join(path_to_image, image_name))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path_to_image, image_name), gray_image)

    if len(gray_image.shape) == 2:
        print(f"{image_name} is gray")
        blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
        canny = cv2.Canny(blur, 10, 50, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=1)
        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 1)

        plt.imshow(rgb)
        plt.show()
    else:
        print(f"{image_name} is not gray")


