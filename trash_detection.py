
import cv2
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================

CAMERA_INDEX = 0
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.35

# Load the pretrained YOLO model
print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

# Open the laptop webcam
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Could not open the camera.")
    print("Try CAMERA_INDEX = 1 if your other camera is available.")
    raise SystemExit

print("Camera started successfully.")
print("Press 'p' to print current detections.")
print("Press 'q' to quit.")

# ==========================================
# 2. OBJECT DETECTION LOOP
# ==========================================

while True:

    # Read one frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read a camera frame.")
        break

    # Run YOLO detection on the current frame
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes and labels
    annotated_frame = result.plot()

    # Store detections from this frame
    current_detections = []

    if result.boxes is not None:

        for box in result.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(
                int, (x1, y1, x2, y2)
            )

            # Class ID and confidence score
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]

            # Calculate the bounding box center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Save this detection
            detection = {
                "class": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y)
            }

            current_detections.append(detection)

            # Draw the center point
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Display center coordinates
            coordinate_text = f"({center_x}, {center_y})"

            cv2.putText(
                annotated_frame,
                coordinate_text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Display the live camera feed
    cv2.imshow("YOLO Trash Detection - Press Q to quit",
               annotated_frame)

    # ==========================================
    # 3. KEYBOARD CONTROLS
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):

        print("\n--- CURRENT FRAME DETECTIONS ---")

        if not current_detections:
            print("No objects detected.")

        for detection in current_detections:
            print(
                f"Object: {detection['class']} | "
                f"Confidence: {detection['confidence']:.2f} | "
                f"Center: {detection['center']} | "
                f"BBox: {detection['bbox']}"
            )

    elif key == ord("q"):
        break

# ==========================================
# 4. RELEASE RESOURCES
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("Detection program stopped.")
