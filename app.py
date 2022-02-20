import cv2
import numpy
import face_recognition
import os
import pickle
from flask import Flask, request, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
import pathlib

UPLOAD_FOLDER = './Students'
ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Gather the names of the students based on the images present in the "Students" folder
pathlib.Path('./Students').mkdir(exist_ok=True)
def StudentImagesPath():
    global StudentImages, StudentNames
    path = './Students'
    StudentImages = []
    StudentNames = []
    StudentList = os.listdir(path)
    for Student in StudentList:
        Img = cv2.imread(f'{path}/{Student}')
        StudentImages.append(Img)
        StudentNames.append(os.path.splitext(Student)[0])  # Get just the names, ignore the .jpg

# Code to decide what the program does
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def start():
    return render_template('start.html')

# Upload student images
@app.route('/upload', methods=['POST','GET'])
def upload():
    return render_template("upload.html")

@app.route('/uploader', methods=['POST'])
def uploader():
    try:
        pic = request.files['pic']
        if pic and allowed_file(pic.filename):
                filename = secure_filename(pic.filename)
                filename=filename.replace("_", " ")
                pic.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filename=os.path.splitext(filename)[0]
        StudentImagesPath()
        return render_template("uploadComplete.html",x=filename)
    except: return render_template("uploadError.html")

# Download attendance file
@app.route('/download')
def downloadFile ():
    try:
        path = "./Attendance.csv"
        return send_file(path, as_attachment=True)
    except: return render_template("downloadError.html")

# Encode any new students
@app.route('/Encode', methods=['POST','GET'])
def encoding():
    try:
        StudentImagesPath()
        x=encode(StudentImages)
        return render_template('encode.html', x=x)
    except: return render_template("encodeError.html")

def encode(images):
    enlist=[]
    with open('encodings', 'wb') as file:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
            en = face_recognition.face_encodings(img)[0]  # Encode the first face in the image
            enlist.append(en) # Append the face data to the list
        pickle.dump(enlist, file) # Save list to file
        return(len(enlist))


# Add names to attendance file
def attendance(name):
    with open('Attendance.csv','a+') as file:
            file.writelines(f'{name}\n')


# Capture video and perform facial recognition
@app.route('/Capture')
def Capture():
    StudentImagesPath()
    if os.path.exists("Attendance.csv"):
        os.remove("Attendance.csv")
    cap = cv2.VideoCapture(0) # Video or image source, 0 for webcam stream,
    # "file path" to open image or video files

    # Open previously saved face encodings
    if (os.path.exists('encodings')):
        with open('encodings', 'rb') as file:
            encodings = pickle.load(file)
    else: return render_template("captureError.html")
    # Continuously scan for faces and match them
    f=0
    namelist=[]
    while True:
        success, img = cap.read()
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if img is None:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)  # Shrink image to improve performance
            # Run face recognition every n'th frame to improve performance
            if f%1==0:
                # Run face rec on shrunk image
                FacesCurFrame = face_recognition.face_locations(imgS)
                EncodesCurFrame = face_recognition.face_encodings(imgS, FacesCurFrame)
                # Find the encoded image with the smallest distance to the image in current frame
                for EncodeFace, FaceLoc in zip(EncodesCurFrame, FacesCurFrame):
                    matches = face_recognition.compare_faces(encodings, EncodeFace)
                    FaceDis = face_recognition.face_distance(encodings, EncodeFace)
                    matchIndex = numpy.argmin(FaceDis)
                    if matches[matchIndex]:
                        name = StudentNames[matchIndex]
                        y1, x2, y2, x1 = FaceLoc
                        x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4  # *4 to undo previous shrinking
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                        namelist.append(name)
        cv2.imshow('Vid', img)
        cv2.waitKey(1)
        f+=1
    cap.release()
    cv2.destroyAllWindows()
    nameset=set(namelist)
    for name in nameset:
            attendance(name)
    return render_template('attendees.html', x=nameset)

@app.route("/contact")
def contact():
    return render_template("contact.html")