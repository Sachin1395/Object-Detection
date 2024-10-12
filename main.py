import cv2
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for,send_file
from werkzeug.utils import secure_filename

def detect_objects(filename):
    # Load YOLO model and the config and weights files
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg.txt")  # Corrected config file path
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    # Get the output layer names of the YOLO model
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load video
    cap = cv2.VideoCapture(filename)  # Use the filename passed to the function

    # Video writer for saving output video (optional)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        h, w = frame.shape[:2]

        # Process each detection in the output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Threshold to filter weak predictions
                    # Get bounding box coordinates
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)
                    x = int(center_x - box_w / 2)
                    y = int(center_y - box_h / 2)

                    # Add to list
                    boxes.append([x, y, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression (NMS) to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes and labels on the frame
        if len(indices) > 0:
            for i in indices:
                if isinstance(i, list) or isinstance(i, np.ndarray):
                    i = i[0]  # Handle list case
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with detected objects
        cv2.imshow("Object Detection", frame)

        # Save the frame to the output video
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

app = Flask(__name__)

# Folder where uploaded videos will be saved
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for video files
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Function to check if the uploaded file is a valid video
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the upload page
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle the video upload and process it
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the uploaded video (pass the uploaded file path)
        detect_objects(file_path)

        return render_template('download.html')
@app.route('/download')
def download_video():
    return send_file('output.avi', as_attachment=True)

if __name__ == "__main__":
    # Make sure the uploads folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)


