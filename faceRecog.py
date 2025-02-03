import numpy as np
import cv2

haar = cv2.CascadeClassifier('haar_faces.xml')

people = ['berra', 'pelin', 'reyhan', 'zeirko', 'irem', 'annem', 'melek']
#features = np.load('features.npy')
#labels = np.load('labels.npy')

facerecog = cv2.face.LBPHFaceRecognizer_create()
facerecog.read('face_trained.yml')

img = cv2.imread(r'/Users/berrasezer/Desktop/face/facevalue/mix/aile/IMG_7464.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('person', gray)

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