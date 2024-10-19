import time
import json
import cv2
from ultralytics import YOLO
from realsense import *

# Load the YOLO model
model = YOLO('yolov8n.pt')
dc = DepthCamera()

# Get the detected objects' names
class_names = model.names

#Get video feed
cap = cv2.VideoCapture("/dev/video4")

while True:
    detections = {"detections": []}
    ret, depth_frame, frame = dc.get_frame()
    if ret:

        # Get frame width to divide into areas
        frame_width = frame.shape[1]
        left_boundary = frame_width / 3
        right_boundary = 2 * frame_width / 3

        results = model(frame)
        annotated_frame = results[0].plot()  # results[0] to access the first image's results

        # Loop through the results to find detected objects
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)  # Get the class index
                object_name = class_names[cls]  # Get the object name using the class index

                # Get the bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0]

                # Calculate the centroid
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                centroid = (int(center_x), int(center_y))

                distance_raw = depth_frame[centroid[1], centroid[0]] / 1000
                distance = round(distance_raw, 1)

                # Determine the location of the object based on the center x-coordinate
                if center_x < left_boundary:
                    location = "on the left"
                elif center_x > right_boundary:
                    location = "on the right"
                else:
                    location = "in front"

                #Structure JSON detection data
                detection_entry = {
                    "name": object_name,
                    "confidence": 1,
                    "position": location,
                    "distance": distance
                }

                # Add the detection entry to the list
                detections["detections"].append(detection_entry)

                # Debugging Print the detected object and its location
                print(f"\n {object_name} detected {location} at {distance} meters")

        with open('detections.json', 'w') as json_file:
            json.dump(detections, json_file, indent=4)

    # Lower the frequency of calculation for better performance
    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
