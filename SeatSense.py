import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pygame
import threading
from collections import deque

class SmallUTKFaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None, limit=5000):
        self.dataset_path = dataset_path
        self.transform = transform
        self.files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")][:limit]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        age, gender, race, _ = file_name.split('_')
        img_path = os.path.join(self.dataset_path, file_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, int(gender)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class ImprovedResNet(nn.Module):
    def __init__(self):
        super(ImprovedResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling and fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)
        
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.dropout(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class VirtualAlarm:
    def __init__(self):
        pygame.mixer.init()
        self.alarm_sound = self._generate_alarm_sound()
        self.is_playing = False
        self.stop_thread = False
        self.alarm_thread = None

    def _generate_alarm_sound(self):
        sample_rate = 44100
        duration = 0.5
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * frequency * t)
        signal = np.int16(signal * 32767)
        stereo_signal = np.column_stack((signal, signal))
        return pygame.sndarray.make_sound(stereo_signal)

    def _play_loop(self):
        while not self.stop_thread:
            self.alarm_sound.play()
            time.sleep(1)

    def start_alarm(self):
        if not self.is_playing:
            self.is_playing = True
            self.stop_thread = False
            self.alarm_thread = threading.Thread(target=self._play_loop)
            self.alarm_thread.start()

    def stop_alarm(self):
        if self.is_playing:
            self.stop_thread = True
            if self.alarm_thread is not None:
                self.alarm_thread.join()
            self.is_playing = False
            pygame.mixer.stop()

class MotionStableClassifier:
    def __init__(self, model, transform, window_size=5, switch_threshold=0.7):
        self.model = model
        self.transform = transform
        self.prediction_history = deque(maxlen=window_size)
        self.switch_threshold = switch_threshold
        self.current_prediction = None
        self.current_confidence = 0.0

    def _apply_motion_stabilization(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _get_smoothed_prediction(self, prediction, confidence):
        self.prediction_history.append((prediction, confidence))
        
        male_count = sum(1 for p, _ in self.prediction_history if p == 0)
        female_count = len(self.prediction_history) - male_count
        
        male_conf = np.mean([conf[0].item() for _, conf in self.prediction_history])
        female_conf = np.mean([conf[1].item() for _, conf in self.prediction_history])
        
        if self.current_prediction is None:
            self.current_prediction = 0 if male_count > female_count else 1
            self.current_confidence = male_conf if self.current_prediction == 0 else female_conf
        else:
            dominant_count = male_count if male_count > female_count else female_count
            dominant_conf = male_conf if male_count > female_count else female_conf
            
            if (dominant_count / len(self.prediction_history) >= 0.6 and 
                dominant_conf > self.switch_threshold):
                self.current_prediction = 0 if male_count > female_count else 1
                self.current_confidence = dominant_conf

        return self.current_prediction, self.current_confidence

    def classify_frame(self, frame):
        stabilized = self._apply_motion_stabilization(frame)
        
        # Convert BGR to RGB and resize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        smooth_class, smooth_conf = self._get_smoothed_prediction(
            predicted_class.item(), 
            probabilities[0]
        )

        return smooth_class, smooth_conf

def train_model(dataset_path, model_save_path='gender_classification_model.pth', force_retrain=False):
    if os.path.exists(model_save_path) and not force_retrain:
        try:
            print("Loading existing model...")
            model = ImprovedResNet()  # Instantiate model before loading
            model.load_state_dict(torch.load(model_save_path))
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Retraining model...")
            os.remove(model_save_path)  # Delete the corrupted or incompatible model file
            # Continue to training as below
    
    print("Training new model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Updated size for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    small_dataset = SmallUTKFaceDataset(dataset_path, transform=transform, limit=4000)
    batch_size = 32
    data_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

    model = ImprovedResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss/len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    print(f"Final model saved to {model_save_path}")
    return model

def capture_and_classify(model, transform, timeout=30):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        start_time = time.time()
        alarm = VirtualAlarm()
        stable_classifier = MotionStableClassifier(model, transform)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predicted_class, confidence = stable_classifier.classify_frame(frame)
            gender_label = "Male" if predicted_class == 0 else "Female"
            
            if gender_label == "Male" and confidence > 0.7:
                alarm.start_alarm()
                frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            else:
                alarm.stop_alarm()
            
            # Add confidence visualization
            bar_length = int(confidence * 200)
            cv2.rectangle(frame, (10, 70), (210, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 70), (10 + bar_length, 90), 
                         (0, 255, 0) if confidence > 0.7 else (0, 255, 255), -1)
            
            cv2.putText(frame, f"Gender: {gender_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Gender Classification', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > timeout:
                break

    except Exception as e:
        print(f"Error during capture and classification: {e}")
    finally:
        alarm.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
    
    return gender_label, confidence

def plot_results(gender, confidence):
    plt.figure(figsize=(8, 8))
    labels = ['Male', 'Female']
    sizes = [confidence if gender == 'Male' else 1 - confidence,
             confidence if gender == 'Female' else 1 - confidence]
    colors = ['lightblue', 'pink']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'Gender Classification Result: {gender}')
    plt.show()

def main():
    dataset_path = 'face dataset/UTKFace'  # Update with your dataset path
    model_save_path = 'gender_classification_model.pth'
    
    # Initialize model
    model = train_model(dataset_path, model_save_path)
    model.eval()
    
    # Updated transform for 224x224 input size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run classification
    gender, confidence = capture_and_classify(model, transform)
    
    print(f"Final Classification: {gender}")
    print(f"Confidence: {confidence:.2f}")
    
    # Plot results
    plot_results(gender, confidence)

if __name__ == '__main__':
    main()