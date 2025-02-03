import numpy as np
import cv2

haar = cv2.CascadeClassifier('haar_faces.xml')

people = [###use the names on trained photos file]

facerecog = cv2.face.LBPHFaceRecognizer_create()
facerecog.read('face_trained.yml')

capture = cv2.VideoCapture(1)

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        print('failed capture')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesrect = haar.detectMultiScale(gray, 1.7, 7)

    for (x, y, w, h) in facesrect:
        faces = gray[y:y+h, x:x+w]

        label, confidence = facerecog.predict(faces)
        print(f'Label = {people[label]} with confidence of {confidence}')

        cv2.putText(frame, str(people[label]), (x, y-50), cv2.FONT_HERSHEY_COMPLEX, 5.0, (0,255,0), 5)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('detected face', frame)

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
