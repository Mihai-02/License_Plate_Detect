import cv2
import numpy as np
import torch
import imutils
import easyocr
import pytesseract
import pandas

from vehicle_detection import detect_vehicles


def perform_ocr(number, ocr_type):
    h, w = number.shape

    new_width = int(3 * w)
    new_height = int(3 * h)
    new_res = (new_width, new_height)
    
    number = cv2.resize(number, new_res, interpolation=cv2.INTER_LINEAR)

    if ocr_type=="easyOCR":
        reader = easyocr.Reader(['en'])
        result = reader.readtext(number)

        text = "__fail__"

        if(result == []):
            return text, 0
        else:
            conf = result[0][-1]*100
            text = result[0][-2]
            if(text == []):
                text = "__fail__"

            return text, conf

    elif ocr_type=="Tesseract":
        #PSM 4 - Single Column of Text of Variable Sizes
        #PSM 6 - Single Uniform Block of Text
        #PSM 7 - Single Text Line
        #PSM 11. Sparse Text
        psm = 7
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        d = pytesseract.image_to_data(number, lang='eng', config=options, output_type=pytesseract.Output.DICT)

        text = ""
        for wd in d['text']:
            if wd!="":
                text = text+wd+" "
        text = text[:-1]

        confidence = d['conf']
        confidence = np.array( [ num for num in confidence if num >= 0 ] )
        conf = confidence.mean()

        return text, conf

def license_plate_detection(image, ocr_type):
    steps_img = []

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps_img.append(gray_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    steps_img.append(blackhat)

    gradX = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    gradY = cv2.Sobel(blackhat, cv2.CV_32F, 0, 1, ksize=3)
    abs_gradX = cv2.convertScaleAbs(gradX)
    abs_gradY = cv2.convertScaleAbs(gradY)
    edge = cv2.addWeighted(abs_gradX, 0.8, abs_gradY, 0.2, 0)
    steps_img.append(edge)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    steps_img.append(edge)

    thresh = cv2.threshold(edge, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    steps_img.append(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    steps_img.append(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    light = cv2.threshold(light, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    steps_img.append(light)
    
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    steps_img.append(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    steps_img.append(thresh)
 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    minAR = 1.3
    maxAR = 4.9

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if ar >= minAR and ar <= maxAR:
            licensePlate = gray_image[y:y + h, x:x + w]

            number = cv2.threshold(licensePlate, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            
            text, conf = perform_ocr(number, ocr_type)

            if text != "__fail__" and len(text)>=4:
                font = cv2.FONT_HERSHEY_SIMPLEX
                res = cv2.putText(image, text=text, org=(x,  y + h + 30), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                res = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0),3)

                steps_img.append(number)
            
                return res, text, conf, steps_img

    return None, "__fail__", 0, steps_img


def lp_detection(initial_img, ocr_type, video=False):
    img_results = []
    nr_results = []
    conf_results = []
    steps_img_2d = []

    if video==False:
        cropped_cars = detect_vehicles(initial_img)    #detectare masini in imaginea initiala
    elif video==True:
        cropped_cars = [initial_img]
    for car in cropped_cars:
        (res, number, confidence, steps_img) = license_plate_detection(car, ocr_type)
        if number !="__fail__" and len(number)>=4:
            img_results.append(res)
            nr_results.append(number)
            conf_results.append(confidence)
            steps_img_2d.append(steps_img)
    
    return img_results, nr_results, conf_results, steps_img_2d