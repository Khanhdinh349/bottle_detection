import cv2
import numpy as np
import joblib
import os

# Load the pre-trained SIFT model
sift_model = joblib.load("sift_trained_model.npy")  # Replace with your path

# Function to predict object class based on the loaded model
def predict_object_class(img, model):
    # Extract SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    if des is not None and len(kp) >= 20:
        des = des[0:20, :]
        vector_data = des.reshape(1, len(des) * len(des[1]))
        
        # Predict the class using the loaded model
        predicted_class = model.predict(vector_data)[0]
        
        # Map the predicted class to the corresponding label (cans, glass, plastic)
        labels = ["cans", "glass", "plastic"]
        predicted_label = labels[predicted_class]
        
        return predicted_label
    else:
        return "Not enough keypoints"

# Create a directory to save images if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# Counter for image filenames
img_counter = 0

# Capture video from the camera
video_capture = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    ret, frame = video_capture.read()

    # Use the SIFT model to predict object class from the camera frame
    predicted_label = predict_object_class(frame, sift_model)
    
    # Display the predicted label on the video frame
    cv2.putText(frame, "Prediction: " + predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the image if the predicted label is "cans"
    if predicted_label == "cans":
        img_name = f"images/cans_img_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved")
        img_counter += 1

    # Display the video frame with predictions
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
