import cv2
from ultralytics import YOLO
import numpy as np

def classify_image(image_path):
    model = YOLO(r"C:\Users\bimap\OneDrive\Documents\Yolov8\best.pt")  # Load the custom model
    results = model(image_path)  # Predict on the image

    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    max_value = probs[0]
    max_index = 0

    for i in range(1, len(probs)):
        if probs[i] > max_value:
            max_value = probs[i]
            max_index = i

    return names_dict[np.argmax(probs)], max_value

def main():
    # OpenCV camera setup
    cap = cv2.VideoCapture(0)  # Use default camera (index 0)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Failed to capture image")
            break

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Classify the captured frame
        classification, probability = classify_image(frame)
        print("Classification:", classification)
        print("Probability:", probability)

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
