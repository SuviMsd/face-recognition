FACE RECOGNITION
This model recognizes your faces lively from webcam, detects it and displays your names.
Remainder: 
I am using windows 11, Pyhton 3.11.
Check whether your laptop"s webcam is working well.

STEPS:
1. Install necessary packages- cv2, os, face_recognition, numpy using the following commands in command prompt.
   For cv2- pip install opencv-python
   For os- pip install os
   For numpy- pip install numpy
   For face_recognition- First install dlib package before installing face recognition.
     To install dlib, download dlib .whl file first which is in my repository.
     pip install dlib
     pip install face_recognition
   Reference (If any errors in installing): 
   https://github.com/ageitgey/face_recognition/issues/175#issue-257710508
2. Create a directory "known_faces" which contains your images as .jpg file
3. Execute train.py file in your IDE( I use PyCharm or Visual Studio Code)
   
     
