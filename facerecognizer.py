import cv2
import face_recognition
import sqlite3


input_video = cv2.VideoCapture(0)
peter_image = face_recognition.load_image_file("Image from iOS.jpg")
peter_face_encoding = face_recognition.face_encodings(peter_image)[0]


bayne_image = face_recognition.load_image_file("IMG_20190807_162611.jpg")
bayne_face_encoding = face_recognition.face_encodings(bayne_image)[0]


faheem_image = face_recognition.load_image_file("faheem.jpg")
faheem_face_encoding = face_recognition.face_encodings(faheem_image)[0]

priya_image = face_recognition.load_image_file("priya.jpg")
priya_face_encoding = face_recognition.face_encodings(priya_image)[0]

rahul_image = face_recognition.load_image_file("rahul.jpg")
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]




known_faces = [
    peter_face_encoding,
    bayne_face_encoding,
    faheem_face_encoding,
    priya_face_encoding,
    rahul_face_encoding
]


face_locations = []
face_encodings = []
face_names = []


def capture_and_encode_face():
    count = 0
    while count < 3:
        ret, frame = input_video.read()
        # Quit when the input video file ends
        if not ret:
            break
        face_encodings = face_recognition.face_encodings(frame)
        known_faces.append(face_encodings)


def evaluate_faces():
    while True:
        # Grab a single frame of video
        ret, frame = input_video.read()

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            if face_encoding is None:
                continue
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            if True in match:
                continue
            else:
                return False


def detect_and_name_face():
    global face_locations, face_encodings, face_names
    while True:
        # Grab a single frame of video
        ret, frame = input_video.read()

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            if face_encoding is None:
                continue
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match[0]:
                name = "Peter"
            elif match[1]:
                name = "Bayne"

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    input_video.release()
    cv2.destroyAllWindows()


detect_and_name_face()
