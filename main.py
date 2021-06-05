from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)
# video_capture = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
ali_image = face_recognition.load_image_file("selfphoto.jpg")
ali_face_encoding = face_recognition.face_encodings(ali_image)[0]

salman_image = face_recognition.load_image_file("salman.jpg")
salman_face_encoding = face_recognition.face_encodings(salman_image)[0]

amirkhan_image = face_recognition.load_image_file("amirkhan.jpg")
amirkhan_face_encoding = face_recognition.face_encodings(amirkhan_image)[0]

shahrukh_image = face_recognition.load_image_file("shahrukh.jpg")
shahrukh_face_encoding = face_recognition.face_encodings(shahrukh_image)[0]

amitabh_image = face_recognition.load_image_file("amitabh.jpg")
amitabh_face_encoding = face_recognition.face_encodings(amitabh_image)[0]

irfankhan_image = face_recognition.load_image_file("irfankhan.jpg")
irfankhan_face_encoding = face_recognition.face_encodings(irfankhan_image)[0]

john_image = face_recognition.load_image_file("john.jpg")
john_face_encoding = face_recognition.face_encodings(john_image)[0]

mahesh_image = face_recognition.load_image_file("mahesh.jpg")
mahesh_face_encoding = face_recognition.face_encodings(mahesh_image)[0]

nawazuddin_image = face_recognition.load_image_file("nawazuddin.jpg")
nawazuddin_face_encoding = face_recognition.face_encodings(nawazuddin_image)[0]

tarique_image = face_recognition.load_image_file("tarique.jpeg")
tarique_face_encoding = face_recognition.face_encodings(tarique_image)[0]

tiger_image = face_recognition.load_image_file("tiger.jpg")
tiger_face_encoding = face_recognition.face_encodings(tiger_image)[0]

varun_image = face_recognition.load_image_file("varun.jpg")
varun_face_encoding = face_recognition.face_encodings(varun_image)[0]

vicky_image = face_recognition.load_image_file("vicky.jpg")
vicky_face_encoding = face_recognition.face_encodings(vicky_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    ali_face_encoding,
    salman_face_encoding,
    amirkhan_face_encoding,
    shahrukh_face_encoding,
    amitabh_face_encoding,
    irfankhan_face_encoding,
    john_face_encoding,
    mahesh_face_encoding,
    nawazuddin_face_encoding,
    tarique_face_encoding,
    tiger_face_encoding,
    varun_face_encoding,
    vicky_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Ajharali Shaikh",
    "salman khan",
    "amir khan",
    "shahrukh khan",
    "amitabh",
    "irfankhan",
    "john",
    "mahesh",
    "nawazuddin",
    "tarique",
    "tiger",
    "varun",
    "vicky"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


def gen_frames():
    # video_capture = cv2.VideoCapture("http://192.168.29.236:8080/video")
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        # cv2.imshow('Video', frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Hit 'q' on the keyboard to quit!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# # Release handle to the webcam
#     video_capture.release()
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

@app.route('/new')
def new():
    # rendering webpage
    return render_template('new.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/delete')
def delete():
    cv2.destroyAllWindows()
    return render_template('index.html')

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='192.168.29.127',port='8080', debug=True)