from ultralytics import YOLO
import cv2
# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")
# Read image
img = cv2.imread("yoloimage.jpeg")
# Perform object detection
results = model(img)
# Draw bounding boxes
output = results[0].plot()

# Display output
cv2.imshow("Object Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
