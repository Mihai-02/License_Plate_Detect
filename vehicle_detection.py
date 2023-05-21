from ultralytics import YOLO
import cv2

def detect_vehicles(img):
    model = YOLO("yolov8l.pt")
    confidence = 0.4
    classes = [2, 3, 5, 7] #2: car; 3: motorcycle; 5: bus; 7: truck

    results = model.predict(img, conf = confidence, classes = classes)
    nr_of_cars = results[0].boxes.data.size()[0]
    all_boxes = results[0].boxes.data.numpy().astype(int)
    
    cropped_cars = []

    for i in range(0, nr_of_cars):      #pentru fiecare masina din imagine
        box = all_boxes[i][:4]
        cropped_cars.append(img[box[1]:box[3], box[0]:box[2]])

    return cropped_cars