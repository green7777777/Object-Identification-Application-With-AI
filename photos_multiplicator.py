import cv2
import os
import random
import numpy as np

DATADIR = "D:/Moje dokumenty/Studia/Semestr 6/WMA/LAB04/Images"
CATEGORIES = ["Banana", "Orange", "Lemon"]

rotation_list = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE, None]


def multiply_images():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        iterator = 11

        for img in os.listdir(path):
            for x in range(20):
                try:
                    print(os.path.join(path, img))
                    rotation_parameter = random.choice(rotation_list)
                    img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2HSV)
                    if rotation_parameter is not None:
                        img_array = cv2.rotate(img_array, rotation_parameter)

                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV).astype("float32")
                    (h, s, v) = cv2.split(img_array)
                    s = s * random.uniform(0.85, 1.15)
                    s = np.clip(s, 0, 255)
                    v = v * random.uniform(0.85, 1.15)
                    v = np.clip(v, 0, 255)
                    img_array = cv2.merge([h, s, v])
                    img_array = cv2.cvtColor(img_array.astype("uint8"), cv2.COLOR_HSV2BGR)

                    img_array = cv2.resize(img_array, (int(img_array.shape[1] * random.uniform(0.75, 1.25)),
                                                       int(img_array.shape[0] * random.uniform(0.75, 1.25))),
                                           interpolation=cv2.INTER_AREA)

                    cv2.imwrite(os.path.join(path, str(iterator) + ".jpeg"), img_array)

                    iterator += 1
                except Exception as e:
                    print(e)


multiply_images()
