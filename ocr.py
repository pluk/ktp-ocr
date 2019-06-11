#!/usr/bin/python
import pytesseract
import cv2
import sys
import numpy as np
import imutils



class ParseError(Exception):
    pass


def process_image(image):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 6))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]

    tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)

    (minVal, maxVal) = (np.min(gradX), np.max(gradX))

    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype('uint8')
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    return thresh


def parse_ktp(image):
    processed_image = process_image(image)

    contours = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only for cv4
    contours = contours[0]

    coord = {}
    locs = []
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # nik
        if 12.50 < ar < 14.85:
            locs.append((x, y, w, h))

    if not locs:
        return

    coord['nik'] = max(locs, key=lambda item: item[3])

    return coord


def process_before_recognize(original, coordinate):

    img = get_image_by_coordinate(original, *coordinate)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = imutils.resize(gray, height=65)

    threshed = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((2, 2), np.float32) / 10
    closed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened


def recognize(image, lang='eng'):
    return pytesseract.image_to_string(image, lang=lang, config='--psm 6')


def get_image_by_coordinate(image, x, y, w, h):
    return image[y - 3:y + h + 3, x - 3:x + w + 3]


def parse_image(data):
    image = np.asarray(bytearray(data), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = resize_image(image)

    coord = parse_ktp(image)

    if coord is None:
        return None

    transformed_nik = process_before_recognize(image, coord['nik'])

    t_nik = recognize(transformed_nik, 'nik')

    return {
        'nik': t_nik
    }


def resize_image(image):
    height, width = image.shape[:2]

    k = 1

    if width >= 1200:
        k = 0.6

    if width > 2000:
        k = 0.3

    if width > 4000:
        k = 0.25

    new_width, new_height = int(k * width), int(k * height)

    if new_height > 900:
        k -= 0.1

    return cv2.resize(image, dsize=None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)


if __name__ == '__main__':
    file_path = sys.argv[1]

    image = cv2.imread(file_path)
    image = resize_image(image)
    print(image.shape[:2])

    coord = parse_ktp(image)

    if coord is None:
        print('NIK not found')
        exit()

    transformed_nik = process_before_recognize(image, coord['nik'])

    t_nik = recognize(transformed_nik, 'nik')

    print('nik: ' + t_nik)

    cv2.imshow('NIK', transformed_nik)
    cv2.waitKey(0)
