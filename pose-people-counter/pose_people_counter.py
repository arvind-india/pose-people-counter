'''
  pose_people_counter

  Quantifies the number of people detected in a camera video stream, pushes
  the data to Firebase


                                   .     .
                                .  |\-^-/|  .
                               /| } O.=.O { |\
                              /´ \ \_ ~ _/ / `\
                            /´ |  \-/ ~ \-/  | `\
                            |   |  /\\ //\  |   |
                             \|\|\/-""-""-\/|/|/
                                     ______/ /
                                     '------'
                       _   _        _  ___
             _ __  ___| |_| |_ _  _| ||   \ _ _ __ _ __ _ ___ _ _
            | '  \/ -_)  _| ' \ || | || |) | '_/ _` / _` / _ \ ' \
            |_|_|_\___|\__|_||_\_, |_||___/|_| \__,_\__, \___/_||_|
                               |__/                 |___/
            -------------------------------------------------------
                            github.com/methylDragon

  Uses OpenCV to capture a video stream, sends the stream to a Tensorflow
  session loaded with a modified ResNet-101 model that outputs data that can
  be used by the algorithm implemented by Eldar for associating detected
  body parts to poses, and hence, countable people.

  The number of people is then pushed to Firebase for real-time logging with
  a partner script. You can also choose to record the annotated video and/or
  images!

  This script uses threading! The video won't lag when inferences are happening!

  Threads:
  1 - Inference
  2 - Video stream input
  3 - Video display

  ---

  Leverages this Human Pose estimation framework entirely:
  https://github.com/eldar/pose-tensorflow

  Additional non-ML related code is all that was added, as well as some
  convenience updates.

  Special Thanks To:
  - Bi Qing (for choosing the model to use)
  - Eldar et al. for the awesome ML model and implementation!

  ---

  License: BSD-2-Clause
'''

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from dataset.pose_dataset import data_to_input
from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut

# Import my custom written OpenCV threading object
from webcam_video_stream import WebcamVideoStream

import pyrebase
import requests
import datetime

import numpy as np
import math
import cv2
import time
import os
from threading import Thread

try:
    import PySpin
except:
    pass

# Start timer
start_time = time.time()

################################################################################
# User configurable parameters
################################################################################

## DEBUGGING AND RECORDING ##

# Debug mode enables verbose information display on the console
# To be honest, I recommend having this on all the time
# The performance loss is not significant
# You only really turn it off if you're doing legit deployment
# and don't want your terminal clogged with print statements
DEBUG = True

# Debug mode for inferences, overrides the normal frequency parameters
# If this is enabled, no data will be logged to Firebase
DEBUG_INFERENCES = True
DEBUG_IMAGE_OUTPUT_FOLDER = "captured_debug_images"

# Record images every set number of seconds
RECORD_IMAGES = True
IMAGE_CAPTURE_FREQUENCY = 1800 # Seconds
IMAGE_OUTPUT_FOLDER = "captured_images"

# Display and/or record the processed video (with annotations)
DISPLAY_VIDEO = True

RECORD_VIDEO = False
VIDEO_OUTPUT_FILENAME = "output.avi"

## Display parameters
DISPLAY_W = 2048 // 4
DISPLAY_H = 1536 // 4
WINDOW_NAME = "Display"

# Frames per second
FPS = 24

## INFERENCE CONFIGURATIONS ##

# How many points a pose needs to have to be considered a person
POINT_THRESHOLD = 5

# Use a Blackfly S Camera
USING_BLACKFLY_S = False

# Set the camera number
CAMERA_NUMBER = 0

# Blackfly 2048x1536
# Logitech 1280x720
CAMERA_W = 1280
CAMERA_H = 720

# Use a canned video instead of a camera feed
USE_CANNED_VIDEO = False
CANNED_VIDEO_PATH = ""

# How often to run the inference
INFERENCE_FREQUENCY = 120 # Seconds

# How much to scale the video that is fed into the CNN (Aim for human height = 360px)
# A higher percentage leads to increased sensitivity, but more false positives
# and longer processing times
SCALING_PERCENTAGE = 100

## FIREBASE ##

PUSH_TO_FIREBASE = True
DEVICE_NAME = "YOUR_DEVICE_LOCATION_HERE"

FIREBASE_CONFIG = {
  "apiKey": "API-KEY-HERE",
  "authDomain": "",
  "databaseURL": "https://YOUR-DATABASE-LINK.firebaseio.com/",
  "storageBucket" : ""
}

################################################################################

# Sanitise the configuration parameters a little bit
IMAGE_CAPTURE_FREQUENCY = int(IMAGE_CAPTURE_FREQUENCY)
DISPLAY_W = int(DISPLAY_W)
DISPLAY_H = int(DISPLAY_H)
POINT_THRESHOLD = int(POINT_THRESHOLD)
INFERENCE_FREQUENCY = int(INFERENCE_FREQUENCY)
SCALING_PERCENTAGE = int(SCALING_PERCENTAGE)

# Reduce FPS if not displaying video to save on CPU usage
if not DISPLAY_VIDEO:
    FPS = 0.2

# If we're in debug mode for inferences, constantly infer and capture
if DEBUG_INFERENCES:
    INFERENCE_FREQUENCY = 5
    IMAGE_OUTPUT_FOLDER = DEBUG_IMAGE_OUTPUT_FOLDER
    IMAGE_CAPTURE_FREQUENCY = 5

################################################################################
# Functions
################################################################################

def rescale(frame, percent=50):
    """Rescale image to target stated percentage."""
    if percent == 100:
        return frame

    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)

    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def color_swapping_yielder():
    """Cycle yielded colors with each call."""
    while True:
        x = yield
        yield (0, 0, x)
        x = yield
        yield (0, x, 0)
        x = yield
        yield (x, x, 0)
        x = yield
        yield (0, x, x)
        x = yield
        yield (x, 0, x)

def get_online_gmt_8_time():
    """ Return current time in GMT +8 as a tuple of (date, time) """
    internet_time = requests.get("http://just-the-time.appspot.com/").text
    date_time = datetime.datetime.strptime(internet_time.strip(), "%Y-%m-%d %H:%M:%S")
    date_time += datetime.timedelta(hours = 8, minutes = 0)

    date = date_time.strftime("%Y-%m-%d")
    time = date_time.strftime("%H:%M:%S")

    return date, time

################################################################################
# Setup
################################################################################

####################
## SET DEBUG MODE ##
####################

# If inference debug mode is engaged, do not push to Firebase
if DEBUG_INFERENCES:
    print("DEBUG_INFERENCES MODE SET: SETTING PUSH_TO_FIREBASE TO FALSE")
    PUSH_TO_FIREBASE = False

###################
## INIT DATABASE ##
###################

if PUSH_TO_FIREBASE:
    # Init Pyrebase app and database
    try:
        firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
        db = firebase.database()
    except:
        print("== FAILED TO INITIALISE FIREBASE ==")
        print("SETTING PUSH_TO_FIREBASE TO FALSE")
        PUSH_TO_FIREBASE = False

###################
## INIT COLOURER ##
###################

# Create yielder object and init the first color
color_tuple = color_swapping_yielder()
next(color_tuple)

#############
## INIT ML ##
#############

# Load ML parameters
cfg = load_config("pose_cfg_multi.yaml")
dataset = create_dataset(cfg)

# Load ML parameters for the spatial model
sm = SpatialModel(cfg)
sm.load()

if DEBUG:
    print("\n== SPATIAL MODEL LOAD OK ==\n")

# Load and setup CNN part detector
# Initializes placeholders for variables, inputs, and outputs
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

if DEBUG:
    print("\n== CNN LOAD OK ==")

########################
## INIT VIDEO STREAMS ##
########################

# Initialise the video stream object
if USE_CANNED_VIDEO:
    cap = WebcamVideoStream(CANNED_VIDEO_PATH, USING_BLACKFLY_S, FPS,
                            CAMERA_W, CAMERA_H, SCALING_PERCENTAGE,
                            DISPLAY_VIDEO, DISPLAY_W, DISPLAY_H, WINDOW_NAME,
                            RECORD_VIDEO, VIDEO_OUTPUT_FILENAME).start()
else:
    cap = WebcamVideoStream(CAMERA_NUMBER, USING_BLACKFLY_S, FPS,
                            CAMERA_W, CAMERA_H, SCALING_PERCENTAGE,
                            DISPLAY_VIDEO, DISPLAY_W, DISPLAY_H, WINDOW_NAME,
                            RECORD_VIDEO, VIDEO_OUTPUT_FILENAME).start()

cv2.putText(cap.overlay_frame, "INITIALISING...",(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(82, 184, 235),4,cv2.LINE_AA)

# Give time for the video stream to initialise
time.sleep(1)

#######################
## INIT VIDEO PARAMS ##
#######################

if USING_BLACKFLY_S:
    width = CAMERA_W * SCALING_PERCENTAGE // 100
    height = CAMERA_H * SCALING_PERCENTAGE // 100
else:
    width = int(cap.get(3)) * SCALING_PERCENTAGE // 100  # float
    height = int(cap.get(4)) * SCALING_PERCENTAGE // 100 # float

if DEBUG:
    print("\n== VIDEO INPUT INITIALISED ==")
    print("WIDTH:", width, "HEIGHT:", height)

###################
## INIT COUNTERS ##
###################

inference_timer = 0
image_capture_timer = 0

###########################
## INIT RECORDING PARAMS ##
###########################

# Make the image capture directory if it doesn't exist
if RECORD_IMAGES:
    # Generate image output path
    IMAGE_OUTPUT_FOLDER = os.path.join(os.path.realpath('.'), IMAGE_OUTPUT_FOLDER)

    try:
        os.mkdir(IMAGE_OUTPUT_FOLDER)
    except:
        print("\nCOULD NOT MAKE FOLDER:", IMAGE_OUTPUT_FOLDER)
        print("IMAGE OUTPUT FOLDER ALREADY EXISTS")

        pass

    print("\nIMAGE OUTPUT FOLDER:", IMAGE_OUTPUT_FOLDER, end="")

# Print sanitised configuration
print("\n\n== POSE PEOPLE COUNTER CONFIGURATION ==",
      "\n\nDEBUG:", DEBUG,
      "\nDEBUG_INFERENCES:", DEBUG_INFERENCES,
      "\n\nRECORD_IMAGES:", RECORD_IMAGES,
      "\nIMAGE_CAPTURE_FREQUENCY:", IMAGE_CAPTURE_FREQUENCY,
      "\nIMAGE_OUTPUT_FOLDER:", IMAGE_OUTPUT_FOLDER,
      "\n\nRECORD_VIDEO:", RECORD_VIDEO,
      "\nVIDEO_OUTPUT_FILENAME:", VIDEO_OUTPUT_FILENAME,
      "\n\nDISPLAY_VIDEO:", DISPLAY_VIDEO,
      "\nDISPLAY_W:", DISPLAY_W,
      "\nDISPLAY_H:", DISPLAY_H,
      "\nFPS:", FPS,
      "\nWINDOW_NAME:", WINDOW_NAME,
      "\n\nPOINT_THRESHOLD:", POINT_THRESHOLD,
      "\nUSING_BLACKFLY_S:", USING_BLACKFLY_S,
      "\nCAMERA_NUMBER:", CAMERA_NUMBER,
      "\nCAMERA_W:", CAMERA_W,
      "\nCAMERA_H:", CAMERA_H,
      "\nUSE_CANNED_VIDEO:", USE_CANNED_VIDEO,
      "\nCANNED_VIDEO_PATH:", CANNED_VIDEO_PATH,
      "\nINFERENCE_FREQUENCY:", INFERENCE_FREQUENCY,
      "\nSCALING_PERCENTAGE:", SCALING_PERCENTAGE,
      "\n\nPUSH_TO_FIREBASE:", PUSH_TO_FIREBASE,
      "\nDEVICE_NAME:", DEVICE_NAME,
      "\n\n===")

################################################################################
# Core loop
################################################################################

# Run the core loop
while(True):
    # If it is time to infer, infer!
    if time.time() - inference_timer > INFERENCE_FREQUENCY:
        # Read the frame from the video stream object
        ret, inference_frame = cap.read()

        # Reset the inference timer
        inference_timer = time.time()

        # If it is time to capture an image, set the image capturing flag to True
        if time.time() - image_capture_timer > IMAGE_CAPTURE_FREQUENCY:
            image_capture_flag = True

            # Reset image capture timer
            image_capture_timer = time.time()
        else:
            image_capture_flag = False


        if DEBUG:
            print("\n\n== INFERRING ==")

            # Display "INFERRING" when calculating inferring
            # The extra code is to center the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "-= INFERRING =-"
            text_size = cv2.getTextSize(text, font, 3, 10)[0]

            # Get coords based on text boundary
            text_X = (width - text_size[0]) // 2
            text_Y = (height + text_size[1]) // 2

            # Add text centered on image
            cv2.putText(cap.overlay_frame, text, (text_X, text_Y ), font, 3, (0, 0, 255), 10)

        # Pre-process the frame for inference
        frame = np.stack((cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY),) * 3, -1)
        image_batch = data_to_input(frame)

        # Compute prediction with the CNN (pull data from output placeholder)
        outputs_np = sess.run(outputs, feed_dict = {inputs: image_batch})

        # Extract parameters from CNN layers
        scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

        # Extract detections and derive pose keypoints
        detections = extract_detections(cfg, scmap, locref, pairwise_diff)
        unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)

        # Output pose keypoints grouped by person
        person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

        # This just counts the number of times you need to iterate!
        people_num = 0
        point_num = 17 # Total no of points on each person
        people_num = person_conf_multi.size / (point_num * 2)
        people_num = int(people_num)
        point_i = 0

        # Reset keypoint counts
        count_past_thresh = 0
        count_below_thresh = 0

        # Clear the persisting overlay from the previous inference
        cap.overlay_frame = np.zeros((height, width, 3), np.uint8)

        # Count keypoints and people!
        for people_i in range(0, people_num):
            point_count = 0
            for point_i in range(0, point_num):
                if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1]) != 0:
                    point_count = point_count +1

            # Set the colors to use
            if people_num <= 25:
                current_color = color_tuple.send(255 - (10 * people_i))
                next(color_tuple)
            else:
                current_color = color_tuple.send(255 - (255 // people_num * people_i))
                next(color_tuple)

            # Draw each point belonging to the current person
            for point_i in range(0, point_num):
                # Recolor the circles based on the person detected
                if (person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1])!= 0:

                    cv2.circle(cap.overlay_frame, (math.floor(person_conf_multi[people_i][point_i][0]),
                               math.floor(person_conf_multi[people_i][point_i][1])),
                               radius = 5, color = current_color, thickness=-1)

                    if RECORD_IMAGES and image_capture_flag:
                        # Annotate saved image frame
                        cv2.circle(inference_frame, (math.floor(person_conf_multi[people_i][point_i][0]),
                                   math.floor(person_conf_multi[people_i][point_i][1])),
                                   radius = 5, color = current_color, thickness=-1)

            # Record the number of detected and suspected people
            # (based off of the threshold)
            if point_count >= POINT_THRESHOLD:
                count_past_thresh += 1
            else:
                count_below_thresh += 1

        # Display text
        cv2.putText(cap.overlay_frame,'People Count: ' + str(count_past_thresh),(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(82, 184, 235),4,cv2.LINE_AA)

        if RECORD_IMAGES and image_capture_flag:
            # Annotate saved image frame
            cv2.putText(inference_frame,'People Count: ' + str(count_past_thresh),(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(82, 184, 235),4,cv2.LINE_AA)

        date_time = get_online_gmt_8_time()

        # Push the relevant people counting data to Firebase
        if PUSH_TO_FIREBASE:
            db.child(DEVICE_NAME).update({"last_update_by_device": {"date": date_time[0], "time": date_time[1]}})
            db.child(DEVICE_NAME).update({"people": count_past_thresh})

        # Record images
        if RECORD_IMAGES and image_capture_flag:
            image_file_name = str(date_time[0]) + "_" + str(date_time[1]) + "-count:" + str(count_past_thresh) + "-" + str(count_below_thresh)

            # Capture image
            cv2.imwrite(os.path.join(IMAGE_OUTPUT_FOLDER, image_file_name + ".jpg"),
                        rescale(inference_frame, 50))

            if PUSH_TO_FIREBASE:
                # Push the relevant image data to Firebase
                db.child(DEVICE_NAME).update({"last_image_captured": {"date": date_time[0], "time": date_time[1], "image_status": "available", "image_file_name": image_file_name + ".jpg"}})

            print("\n== IMAGE CAPTURED:", image_file_name, "==\n")

        if DEBUG:
            print("DETECTED:", count_past_thresh, "+ SUSPECTED:", count_below_thresh)
            print()

    # Visualise waiting time with a line
    if DISPLAY_VIDEO:
        cv2.line(cap.overlay_frame, (width, height), (width, height - int(((time.time() - inference_timer) / INFERENCE_FREQUENCY) * height) - 10), (82, 184, 235), 20)

    # Print status message
    if DEBUG:
        print("Seconds waited: "
              + '{:.1f}'.format(time.time() - inference_timer)
              + "/"
              + str(INFERENCE_FREQUENCY)
              + " | Uptime: "
              + str(round(time.time() - start_time, 1))
              + " | Video capture status: "
              + str(ret), end="\r")

    # If the capture is ever stopped, break
    # The capture object implements the listener for 'q'
    if cap.stopped:
        break

    time.sleep(.05)

# Cleanup
cap.stop()
cv2.destroyAllWindows()
