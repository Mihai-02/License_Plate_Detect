import cv2
import numpy as np
import torch
from ocrmodel import CNNModel
from PIL import Image
import os

def add_margin_numpy(img_np, top, right, bottom, left, color):
    pil_img = Image.fromarray(img_np)

    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return np.array(result)

def segment_characters(plate_image, valley_thresh):
    steps_img = []

    vertical_projection = np.sum(plate_image, axis=0)
    
    # Visualize the projection profile
    projection_viz = np.zeros((100, len(vertical_projection)), dtype=np.uint8)
    for i, sum_val in enumerate(vertical_projection):
        # Normalize the sum value to fit in our visualization
        height = int((sum_val / np.max(vertical_projection)) * 100)
        cv2.line(projection_viz, (i, 100), (i, 100-height), 255, 1)

    steps_img.append(projection_viz)
    
    valleys = []
    min_gap_width = 1
    in_valley = False
    valley_start = 0

    min_value = np.min(vertical_projection)
    
    threshold = (valley_thresh / 100.0) * min_value

    viz_image = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(vertical_projection)):
        if vertical_projection[i] < threshold:
            if not in_valley:
                valley_start = i
                in_valley = True
                # Draw valley start line in red
                cv2.line(viz_image, (i, 0), (i, plate_image.shape[0]), (0,0,255), 1)
        else:
            if in_valley:
                valley_end = i
                if valley_end - valley_start >= min_gap_width:
                    valleys.append((valley_start, valley_end))
                    # Draw valley end line in green
                    cv2.line(viz_image, (i, 0), (i, plate_image.shape[0]), (0,255,0), 1)
                in_valley = False

    steps_img.append(viz_image)
    
    print("Number of valleys found:", len(valleys))
    
    return valleys, steps_img



def manual_ocr(plate_image, roi_nr, valley_thresh):
    height = 70
    ratio = height / plate_image.shape[0]
    width = int(plate_image.shape[1] * ratio)
    plate_image = cv2.resize(plate_image, (width, height))

    valleys, steps_img = segment_characters(plate_image, valley_thresh)

    if len(valleys)<=2:
        return "__fail__", []
    
    os.makedirs("individual_characters", exist_ok=True)

    char_candidates = []
    for i in range(1, len(valleys)):  # Skip first valley (background)
        valley_start, valley_end = valleys[i-1], valleys[i]
        char_image = plate_image[:, valley_start[1]-3:valley_end[0]+3]
        char_candidates.append(char_image)

    last_valley_end = valleys[-1][1]
    char_image_last = plate_image[:, last_valley_end:-5]  # Extract last character until the image end
    char_candidates.append(char_image_last)

    classes = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    text = ""
    ocr_model = torch.load("../ocr_training/model_new_dataset_2_85x45.pth")
    

    for idx, char_img in enumerate(char_candidates):
        if char_img.shape[0]<=0 or char_img.shape[1]<=0:
            continue

        height, width = char_img.shape
        # Mask out the top and bottom regions - usually containing unnecessary information, as plate borders
        char_img[:int(height/8), :] = 0
        char_img[-int(height/10):, :] = 0

        char_img[:, :int(width/10)] = 0
        char_img[:, -int(width/10):] = 0 

        char_img = cv2.resize(char_img, dsize=(35, 75), interpolation=cv2.INTER_LINEAR)
        char_img = add_margin_numpy(char_img, 5, 5, 5, 5, (0))
        
        char_img = cv2.threshold(char_img, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(f"individual_characters/char_{roi_nr}_{idx}.jpg", char_img)

        char_img = char_img / 255.0  

        char_img = torch.Tensor(char_img).unsqueeze(0).unsqueeze(0)
        char_img = char_img.to('cuda')

        outputs = ocr_model(char_img)

        predicted_class = torch.argmax(outputs, dim=1).item()
        if classes[predicted_class] != '_':
            text = text + str(classes[predicted_class])

    return text, steps_img


