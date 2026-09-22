import cv2
import face_recognition
# Load known image
known_image = face_recognition.load_image_file("known.jpg")
# Generate face encoding
known_encoding = face_recognition.face_encodings(known_image)[0]
# Open webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect faces
    face_locations = face_recognition.face_locations(rgb)
    # Generate encodings
    face_encodings = face_recognition.face_encodings(
        rgb,
        face_locations
    )
    # Compare faces
    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):
        matches = face_recognition.compare_faces(
            [known_encoding],
            face_encoding
        )
        name = "Unknown"
        if matches[0]:
            name = "Nilesh"
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )
        cv2.putText(
            frame,
            name,
            (left, top-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
