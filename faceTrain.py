import os
import cv2
import numpy as np

#p = []
#for i in os.listdir(r'/Users/berrasezer/Desktop/gettingstarted/testface/face/facetrain'):
#    p.append(i)
#print(p)

people = ['berra', 'pelin', 'reyhan', 'zeirko', 'irem', 'annem', 'melek']
DIR = r'/Users/berrasezer/Desktop/face/facetrain'

haar = cv2.CascadeClassifier('haar_faces.xml')

features = []
labels = []

def createTrain():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):

            if img.startswith('.'): # to delete .ds_store files!!! look what are those???
                continue
            
            imgpath = os.path.join(path, img)

#            if not os.path.exists(imgpath):
#                print(f"Error: File does not exist at {imgpath}")
#                continue

#            print(f"Loading image from {imgpath}")
            imgarray = cv2.imread(imgpath)
#            if imgarray is None:
#                print(f"Error: Could not load image at {imgpath}")
#                continue

            gray = cv2.cvtColor(imgarray, cv2.COLOR_BGR2GRAY)

            facerect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in facerect:
                face = gray[y:y+h, x:x+w]

                features.append(face)
                labels.append(label)

createTrain()
print('training done!!!')

#print(f'Length of the features = {len(features)}')
#print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

facerecog = cv2.face.LBPHFaceRecognizer_create()

facerecog.train(features, labels)

facerecog.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)






                




