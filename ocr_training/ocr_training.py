import numpy as np
import os
from PIL import Image, ImageFilter
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments

from torch.utils.data import Dataset, DataLoader
import random

from ocrmodel import CNNModel

def add_strong_gaussian_noise(img, mean=0, std=20):
    img = np.array(img)
    noise = np.random.normal(mean, std, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)  # Clipping to valid range
    return Image.fromarray(img.astype(np.uint8))

def apply_strong_blur(img, radius=5):
    return img.filter(ImageFilter.GaussianBlur(radius))

def convert_to_no_char(img, label, p=0.1):
    if random.random() < p:
        # Apply strong corruption like noise or blur
        img = add_strong_gaussian_noise(img)
        img = apply_strong_blur(img, radius=10)
        one_hot = np.zeros(38)  # Create a zero vector of length `num_classes`
        one_hot[37] = 1  # Set the index corresponding to the label to 1
        label = torch.tensor(one_hot, dtype=torch.float32)  # Return as a tensor
    return img, label

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx][0])

        image, label_np = convert_to_no_char(image, self.labels[idx], p=0.016)

        image = torchvision.transforms.ToTensor()(image)

        if isinstance(label_np, torch.Tensor):
            label = label_np.detach()
        else:
            label = torch.tensor(label_np, dtype=torch.float32).detach()

        return image, label
    
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch]).float()
    labels = torch.stack([item[1] for item in batch]).long()

    return {"x": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    labels = np.argmax(labels, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# classes = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'  # the representation of the classes

dataset=[]
labels=[]
count=0
folders=os.listdir(r"Dataset/license_plate_chars_new/")
for i in folders:
    print(i)
    for j in os.listdir(r"Dataset/license_plate_chars_new/"+str(i)):
        image = Image.open(r"../ocr_training/Dataset/license_plate_chars_new/"+str(i)+'/'+str(j))
        dataset.append(np.array([   np.asarray(image)/255.0 ]  ))
        labels.append(count)
    count+=1

labels = np.array(labels)
labels = labels.reshape(-1, 1)

# Initialize the OneHotEncoder
type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
type_encoder.fit(np.arange(38).reshape(-1, 1))  # Fit to labels from 0 to 37 (37: not a char)

# One-hot encode the labels
labels = type_encoder.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state=42)

train_dataset = CustomDataset(images=X_train, labels=y_train)
valid_dataset = CustomDataset(images=X_test, labels=y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

model = CNNModel()

training_args = TrainingArguments(
    output_dir="./results_model", 
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=8e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    weight_decay=0.02,
    logging_dir="./logs_model",
    logging_steps=1000,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)

trainer.train()

torch.save(model, "model_85x45.pth")