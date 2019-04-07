from imutils import paths
import face_recognition
import pickle
import cv2
import os

def encodings():
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images('dataSet'))

    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):


        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb,model='hog')

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
          knownEncodings.append(encoding)
          knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open('encodings.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()


def recognizer():
    print("[INFO] loading encodings...")
    data = pickle.loads(open('encodings.pickle', "rb").read())

    image = (cv2.imread('test.jpg'))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.2)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    imageS = cv2.resize(image, (1920, 1080))
    cv2.imshow("Image", imageS)
    cv2.waitKey(0)

encodings()
recognizer()