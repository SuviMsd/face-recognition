import cv2
import face_recognition
import os
import numpy as np

# Load known faces and their names from a directory
known_face_encodings = []
known_face_names = []

# Directory containing known faces
known_faces_dir = "known_faces"

# Iterate over files in the directory
for filename in os.listdir(known_faces_dir):
    # Load the image
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    # Extract face encoding
    face_encoding = face_recognition.face_encodings(image)[0]
    # Extract name from the filename (assuming filename is in the format "name.jpg")
    name = os.path.splitext(filename)[0]
    # Append encoding and name to lists
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame from BGR to RGB (as face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize list to store names for current frame
    names = []

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # Calculate Euclidean distance between face encoding and known face encodings
        distances = np.linalg.norm(np.array(known_face_encodings) - np.array(face_encoding), axis=1)
        # Assign name of the known face with smallest distance
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < 0.45:  # Adjust threshold as needed
            name = known_face_names[min_distance_index]
        else:
            name = "Unknown"
        names.append(name)

    # Draw rectangles and display names for each face found in the frame
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
