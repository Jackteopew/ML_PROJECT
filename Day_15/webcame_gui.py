import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from PIL import Image, ImageTk
import pygame  # For audio playback
import cv2  # For webcam support
import csv

# Initialize GUI
win = tk.Tk()
win.title("Plant Disease Detection System")
win.geometry("600x700")
win.config(bg="#f0f8ff")

# Title Label
title_label = Label(win, text="Plant Disease Detection System", font=("Arial", 20, "bold"), bg="#f0f8ff", fg="#333")
title_label.pack(pady=10)

# Create Image Display Area
img_label = Label(win, bg="#ffffff", relief="solid", bd=1)
img_label.pack(pady=10)

# Create Text Display Area for Results
result_label = Label(win, text="", fg='black', font=("Arial", 14), bg="#f0f8ff")
result_label.pack(pady=10)

# Prediction History Box
history_label = Label(win, text="Prediction History", font=("Arial", 12, "bold"), bg="#f0f8ff")
history_label.pack(pady=10)

history_box = Text(win, height=10, width=50, font=("Arial", 10))
history_box.pack(pady=10)

# Fixed Frame for Buttons
button_frame = Frame(win, bg="#f0f8ff")
button_frame.pack(pady=20)

# Initialize pygame mixer
pygame.mixer.init()

# Load Model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")

# Define Labels
label = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
         "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
         "Corn_(maize)___Healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
         "Grape___Esca_(Black_Measles)", "Grape___Healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
         "Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight", "Tomato___Bacterial_spot",
         "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
         "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
         "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]

# Prediction History
history = []

# Function to Export History
def export_history():
    with open('prediction_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Status", "Label", "Confidence (%)"])
        writer.writerows(history)
    print("History exported successfully!")

# Function to Process Image
def process_image(path2):
    try:
        # Display the selected image in the GUI
        img = Image.open(path2)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)
        img_label.image = img

        # Preprocess the Image
        test_image = image.load_img(path2, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make Prediction
        result = loaded_model.predict(test_image)
        confidence = np.max(result) * 100
        label2 = label[result.argmax()]

        # Determine Status: Healthy or Diseased
        if 'healthy' in label2.lower():
            status = "Healthy"
            audio_file = 'healthy.mp3'
            result_label.configure(fg='green')  # Green for healthy
        else:
            status = "Diseased"
            audio_file = 'disease.mp3'
            result_label.configure(fg='red')  # Red for diseased

        # Play audio file
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Display Prediction and Accuracy
        prediction_text = f"Status: {status}\nLabel: {label2}\nConfidence: {confidence:.2f}%"
        result_label.configure(text=prediction_text)
        print(prediction_text)

        # Update History
        history.append([status, label2, f"{confidence:.2f}"])
        history_box.insert(END, prediction_text + "\n")

    except Exception as e:
        print(f"Error: {str(e)}")

# Function for Webcam Capture
def webcam_capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam - Press SPACE to Capture', frame)

        # Press 'space bar' to capture and exit
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite('captured_image.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Process the captured image
    process_image('captured_image.jpg')

# Add Buttons to Fixed Frame
btn_webcam = Button(button_frame, text="Use Webcam", command=webcam_capture, font=("Arial", 12, "bold"), bg="#28A745", fg="white", padx=20, pady=10)
btn_webcam.pack(side=LEFT, padx=10)

btn_export = Button(button_frame, text="Export History", command=export_history, font=("Arial", 12, "bold"), bg="#FFC107", fg="black", padx=20, pady=10)
btn_export.pack(side=LEFT, padx=10)

# Start GUI
win.mainloop()
