#Installing Packages 

import cv2
import numpy as np
import face_recognition as fr # This package is use for FACE RECOGNITION

#Before installing Face Recognition package You have to install dlib package


#Below function use for upload the face images and encoding the images
def LoadFaces():
    
    mayur_image = fr.load_image_file("Mayur.jpeg")
    mayur_face_encoding = fr.face_encodings(mayur_image)[0]
    
    elon_image = fr.load_image_file("Elon_Musk.jpeg")
    elon_face_encoding = fr.face_encodings(elon_image)[0]
    
    # This block store the encoding faces
    
    known_face_encondings = [
        mayur_face_encoding,
        elon_face_encoding
    ]
    
    # This block store the name of encoding faces
    
    known_face_names = [
        "Mayur",
        "Harshal",
        "Elon Musk"
    ]
    
    return known_face_encondings, known_face_names;

known_face_encondings, known_face_names = LoadFaces()

cap = cv2.VideoCapture(0) 

ret, frame = cap.read()
ret, frame1 = cap.read()

while cap.isOpened():
    
    # This block is use for Face Recognition
    
    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
     
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # This block is use for Motion Detection 
    
    diff = cv2.absdiff(frame, frame1)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < 900:
            continue
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Status: {}".format('Motion Detected'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imshow("Face Recognition", frame)
    frame = frame1
    ret, frame1 = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  #To end the frame
        break
    
cv2.destroyAllWindows()
cap.release()