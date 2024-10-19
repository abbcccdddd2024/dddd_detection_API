import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r'C:\ABBCCC\yolov8n.pt')

# Start video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For webcam use 0, or replace with 'video.mp4' for video file

# Set desired resolution (Width, Height)
desired_width = 1280 // 2  # e.g., 1280 for HD resolution
desired_height = 720 // 2   # e.g., 720 for HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from video/camera
    if not ret:
        print("Failed to grab frame")
        break

    # Get image dimensions
    frame_height, frame_width = frame.shape[:2]
    left_bound = frame_width // 3
    right_bound = 2 * frame_width // 3

    # Split the frame into left, middle, and right parts
    left_roi = frame[:, :left_bound]
    middle_roi = frame[:, left_bound:right_bound]
    right_roi = frame[:, right_bound:]

    # Apply Canny edge detection to the left and right parts
    left_canny = cv2.Canny(left_roi, 80, 250)
    right_canny = cv2.Canny(right_roi, 80, 250)

    # Convert the Canny edge-detected images back to 3-channel BGR format
    left_canny_bgr = cv2.cvtColor(left_canny, cv2.COLOR_GRAY2BGR)
    right_canny_bgr = cv2.cvtColor(right_canny, cv2.COLOR_GRAY2BGR)

    # Reassemble the frame: Canny on left and right, RGB in the middle
    combined_frame = cv2.hconcat([left_canny_bgr, middle_roi, right_canny_bgr])

    # Object detection only on the left and right regions
    detected_objects_left = []
    detected_objects_right = []

    # YOLO model only processes left and right ROIs (not middle)
    left_results = model(left_roi)  # Object detection for left side
    right_results = model(right_roi)  # Object detection for right side

    # Process left-side detection results
    for r in left_results:
        for obj in r.boxes:
            class_id = int(obj.cls[0])  # Get the class ID
            object_name = model.names[class_id]  # Get object name from class ID

            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Coordinates of the bounding box
            detected_objects_left.append(object_name)

            # Draw the bounding box on the left Canny frame
            cv2.rectangle(left_canny_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left_canny_bgr, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Process right-side detection results
    for r in right_results:
        for obj in r.boxes:
            class_id = int(obj.cls[0])  # Get the class ID
            object_name = model.names[class_id]  # Get object name from class ID

            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Coordinates of the bounding box
            detected_objects_right.append(object_name)

            # Draw the bounding box on the right Canny frame
            cv2.rectangle(right_canny_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(right_canny_bgr, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print detected objects on each side
    if detected_objects_left:
        print(f"Detected objects on left: {', '.join(detected_objects_left)}")
    if detected_objects_right:
        print(f"Detected objects on right: {', '.join(detected_objects_right)}")

    # Reassemble the frame: Canny with bounding boxes on the left and right, and RGB in the middle
    combined_frame = cv2.hconcat([left_canny_bgr, middle_roi, right_canny_bgr])

    # Show the combined frame (left and right with Canny and detection, middle in RGB)
    cv2.imshow("YOLO Detection on Left/Right with Canny", combined_frame)

    # Write the frame to the output video file


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object, video writer, and close display window
cap.release()
cv2.destroyAllWindows()
