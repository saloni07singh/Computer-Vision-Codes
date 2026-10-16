import cv2

# Open camera 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

# Read first frame
ret, frame = cap.read()

if not ret or frame is None:
    print("Could not read the first frame")
    cap.release()
    exit()

# Select object to track
print("Draw a rectangle around the object.")
print("Press ENTER or SPACE to confirm.")
print("Press C to cancel.")

bbox = cv2.selectROI(
    "Select Object",
    frame,
    False
)

cv2.destroyWindow("Select Object")

# Check if an object was selected
x, y, w, h = [int(v) for v in bbox]

if w == 0 or h == 0:
    print("No object was selected")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Create CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Initialize tracker
tracker.init(frame, bbox)

print("Tracking started!")
print("Press Q to quit.")

while True:

    # Read live camera frame
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Could not read frame")
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    if success:

        x, y, w, h = [int(v) for v in bbox]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display tracking status
        cv2.putText(
            frame,
            "Tracking",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Tracking Failed",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Display live video
    cv2.imshow(
        "Live Object Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()