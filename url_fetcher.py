import urllib.request
import cv2 as cv2
import numpy as np
import os

def store_raw_images():
    neg_images_link = ''
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    neg_images_dir = 'neg'
    if not os.path.exists(neg_images_dir):
        os.makedirs(neg_images_dir)

    pic_num = 1

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/" + str(pic_num) + '.jpg')
            img = cv2.imread(i, "neg/" + str(pic_num) + '.jpg', cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (100,100))

        except Exception as e:
            print(str(e))
