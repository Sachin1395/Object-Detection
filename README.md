# Object Detection in Videos and Images Using YOLO
# Overview
This project implements real-time object detection in videos and images using the YOLO (You Only Look Once) deep learning model. Users can upload videos or images via a web interface built with Flask, and the application processes the uploaded files to detect various objects, drawing bounding boxes and labels around them.

# Download weights
Download the YOLOv4 weights from [Google Drive](https://drive.google.com/drive/folders/1LugY68ppn7SBtvSOTkiowqz4eFDhYeTg?usp=drive_link).

# Features
Real-time Object Detection: Utilizes the YOLOv4 model for fast and accurate detection of objects in both videos and images.
Web-based Interface: Built with Flask, allowing users to easily upload and process files from their browser.
Video and Image Support: The application can process both video files (e.g., MP4, AVI) and static images (e.g., JPG, PNG).
Download Processed Output: Users can download the processed video or image with detected objects.

# Requirements
Python 3.x
Flask
OpenCV
NumPy

# Code Structure
app.py: The main Flask application handling the upload and processing of videos/images.
templates/: Contains HTML templates for the web interface.
uploads/: Directory to store uploaded files.
output.avi: Default output file for processed videos.
output_image.jpg: Default output file for processed images.
Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

# Acknowledgments
YOLO for the object detection model.
Flask for the web framework.
OpenCV for computer vision functionality.
