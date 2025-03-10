import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import os

class AddHorizontalLine:
    def __init__(self, thickness_range=(2, 15), prob=0.2):
        self.thickness_range = thickness_range
        self.prob = prob
    
    def __call__(self, img_array):
        if random.random() > self.prob:
            return img_array
            
        height, width = img_array.shape
        thickness = random.randint(self.thickness_range[0], self.thickness_range[1])
        length = width
        
        case = random.choice(['top', 'bottom', 'both'])
        
        if case == 'top' or case == 'both':
            y_pos_top = random.randint(0, thickness)
            img_array[y_pos_top:y_pos_top + thickness, 0:length] = 255
        
        if case == 'bottom' or case == 'both':
            y_pos_bottom = random.randint(height - thickness, height)
            img_array[y_pos_bottom:y_pos_bottom + thickness, 0:length] = 255
        
        return img_array
    

def create_character_image(char, fonts, width=85, height=45):
    pad_factor = 1.2
    img_width = int(width * pad_factor)
    img_height = int(height * pad_factor)
    img = Image.new('L', (img_width, img_height), color=0)
    draw = ImageDraw.Draw(img)
    
    # Randomly select font
    font_path = random.choice(fonts)
    # Make character height 40-85% of image height
    font_size = int(height * random.uniform(0.4, 0.85))
    font = ImageFont.truetype(font_path, font_size)
    
    # Get character size for centering
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center character
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2
    
    # Draw white character
    draw.text((x, y), char, fill=255, font=font)

    img_array = np.array(img)
    
    # Apply rotation
    if random.random() < 0.7:
        angle = random.uniform(-20, 20)
        center = (img_width // 2, img_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_array = cv2.warpAffine(img_array, rotation_matrix, (img_width, img_height))
    
    # Apply perspective transform
    if random.random() < 0.4:
        offset_range = (-width//6, width//6)
        pts1 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        pts2 = np.float32([
            [random.uniform(*offset_range), random.uniform(*offset_range)],
            [img_width - random.uniform(*offset_range), random.uniform(*offset_range)],
            [random.uniform(*offset_range), img_height - random.uniform(*offset_range)],
            [img_width - random.uniform(*offset_range), img_height - random.uniform(*offset_range)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(img_array, matrix, (img_width, img_height))
    
    # Resize to final size
    img_array = cv2.resize(img_array, (width, height))
    
    # Convert to binary
    _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    
    # Add horizontal line artifacts
    line_adder = AddHorizontalLine()
    img_array = line_adder(img_array)
    
    # Add salt and pepper noise
    if random.random() < 0.1:
        prob_noise = random.uniform(0.01, 0.02)
        mask = np.random.random(img_array.shape) < prob_noise
        img_array[mask] = random.choice([0, 255])
    
    return img_array

def generate_dataset(output_dir, chars, fonts, samples_per_char=1000, width=45, height=85):
    os.makedirs(output_dir, exist_ok=True)
    
    for char in chars:
        char_dir = os.path.join(output_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        
        for i in range(samples_per_char):
            img_array = create_character_image(char, fonts, width, height)
            
            img_path = os.path.join(char_dir, f'{char}_{i}.png')
            cv2.imwrite(img_path, img_array)


chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
fonts = []

for j in os.listdir(r"Dataset/fonts"):
    fonts.append("Dataset/fonts/" + str(j))

generate_dataset('license_plate_chars_new', chars, fonts, samples_per_char=2000)