# coding: utf8

"""
Description :
Convert MIT image data to the standard we need.
"""

import os
import cv2


def main():
    path1 = 'cars128x128//'
    path2 = 'pedestrians128x64//'
    path3 = 'data//train//cars//'
    path4 = 'data//train//pedestrians//'

    for root, dirs, files in os.walk(path1):
        for f in files:
            n = f.split('.')[0]
            img = path1 + f
            image = cv2.imread(img)
            resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path3 + str(n) + '.jpg', resized_image)

    for root, dirs, files in os.walk(path2):
        for f in files:
            n = f.split('.')[0]
            img = path2 + f
            image = cv2.imread(img)
            resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path4 + str(n) + '.jpg', resized_image)


if __name__ == '__main__':
    main()
