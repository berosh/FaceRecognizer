import os
import cv2
import numpy as np

people = [###use the names on trained photo files]
DIR = r'###path to trained photos file'

haar = cv2.CascadeClassifier('haar_faces.xml')

features = []
labels = []

def createTrain():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):

            if img.startswith('.'):
                continue
            
            imgpath = os.path.join(path, img)

            imgarray = cv2.imread(imgpath)

            gray = cv2.cvtColor(imgarray, cv2.COLOR_BGR2GRAY)

            facerect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in facerect:
                face = gray[y:y+h, x:x+w]

                features.append(face)
                labels.append(label)

createTrain()
print('training done!!!')

features = np.array(features, dtype='object')
labels = np.array(labels)

facerecog = cv2.face.LBPHFaceRecognizer_create()

facerecog.train(features, labels)

facerecog.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)






                




