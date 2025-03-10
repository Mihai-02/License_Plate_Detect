import cv2
import imutils
import pytesseract
import torch
from vehicle_detection import detect_vehicles
from ocrmodel import CNNModel
from edge_detection import detect_edges
from ocr import manual_ocr

def perform_ocr(number, ocr_type, roi_nr, valley_thresh):
    if ocr_type=="CustomCNN":
        text, steps_img = manual_ocr(number, roi_nr, valley_thresh)

        return text, steps_img

    elif ocr_type=="Tesseract":
        #PSM 4 - Single Column of Text of Variable Sizes
        #PSM 6 - Single Uniform Block of Text
        #PSM 7 - Single Text Line
        #PSM 11. Sparse Text
        psm = 6
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)

        h, w = number.shape

        new_width = int(3 * w)
        new_height = int(3 * h)
        new_res = (new_width, new_height)

        number = cv2.resize(number, new_res, interpolation=cv2.INTER_LINEAR)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        d = pytesseract.image_to_data(number, lang='eng', config=options, output_type=pytesseract.Output.DICT)

        text = ""
        for wd in d['text']:
            if wd!="":
                text = text+wd+" "
        text = text[:-1]

        steps_img = []

        return text, steps_img

def license_plate_detection(image, ocr_type, valley_thresh):
    steps_img = []

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps_img.append(gray_image)

    blackhat = gray_image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    steps_img.append(blackhat)

    edge = detect_edges(blackhat)
    steps_img.append(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    steps_img.append(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.erode(edge, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edge = cv2.dilate(edge, kernel, iterations=2)
    steps_img.append(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    steps_img.append(light)

    edge = cv2.bitwise_and(edge, edge, mask=light)
    steps_img.append(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.dilate(edge, kernel, iterations=2)
    edge = cv2.erode(edge, kernel, iterations=2)

    edge = cv2.threshold(edge, 127, 255,cv2.THRESH_BINARY)[1]

    steps_img.append(edge)

    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    minAR = 1.5
    maxAR = 5.9

    for idx, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if ar >= minAR and ar <= maxAR:
            licensePlate = gray_image[y:y + h, x:x + w]

            number = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

            text, steps_ocr = perform_ocr(number, ocr_type, idx, valley_thresh)

            if text != "__fail__" and len(text)>=4:
                font = cv2.FONT_HERSHEY_SIMPLEX
                res = cv2.putText(image, text=text, org=(x,  y + h + 30), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                res = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0),3)

                steps_img.append(number)

                for step in steps_ocr:
                    steps_img.append(step)

                return res, text, steps_img

    return None, "__fail__", steps_img


def lp_detection(initial_img, ocr_type, valley_thresh=1, video=False):
    img_results = []
    nr_results = []
    steps_img_2d = []

    if video==False:
        cropped_cars = detect_vehicles(initial_img)
    elif video==True:
        cropped_cars = [initial_img]
    for car in cropped_cars:
        (res, number, steps_img) = license_plate_detection(car, ocr_type, valley_thresh)
        if number !="__fail__" and len(number)>=3:
            img_results.append(res)
            nr_results.append(number)
            steps_img_2d.append(steps_img)

    return img_results, nr_results, steps_img_2d