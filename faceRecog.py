import numpy as np
import cv2

haar = cv2.CascadeClassifier('haar_faces.xml')

people = [###use the names on the trained photos file]

facerecog = cv2.face.LBPHFaceRecognizer_create()
facerecog.read('face_trained.yml')

img = cv2.imread(r'###path to one of the photos in valued photos file')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

facesrect = haar.detectMultiScale(gray, 1.7, 7)

for (x, y, w, h) in facesrect:
    faces = gray[y:y+h, x:x+w]

    label, confidence = facerecog.predict(faces)
    print(f'Label = {people[label]} with confidence of {confidence}')

    cv2.putText(img, str(people[label]), (x, y-50), cv2.FONT_HERSHEY_COMPLEX, 5.0, (0,255,0), 5)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('detected face', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
