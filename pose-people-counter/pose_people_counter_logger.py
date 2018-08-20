# -*- coding: utf-8 -*-
'''
  pose_people_counter_logger

  Pulls people-counter data from Firebase and logs to .csv (Logs failures!)

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

  Pulls periodically from the stated Firebase data-base, and logs the data
  to a .csv, wiping the data to the Firebase entries that were pulled.

  If an entry is not refilled by the time this logger checks again, a fail
  state is reflected on the database.

  The .csv file that is written to is appended. If a new file is created,
  it is initialised with the column legend on the first row.

  The current date and time of the logger is used to associate times with
  the people count data. On the first run-through of the script though,
  no data is saved.

REQUIRES
--------
- Internet access!

'''

import pyrebase
import requests

import csv
import datetime
import time
import socket

import os

################################################################################
# User configurable parameters
################################################################################

# Firebase credentials
FIREBASE_CONFIG = {
  "apiKey": "API-KEY-HERE",
  "authDomain": "",
  "databaseURL": "https://YOUR-DATABASE-LINK.firebaseio.com/",
  "storageBucket" : ""
}

LOG_FILE_NAME = "pose_people_counting_log"

# In increments of 10s (30 here is 300 seconds -> 5 minutes)
LOG_FREQUENCY = 30

################################################################################
# Functions
################################################################################

def get_online_gmt_8_time():
    """Return current time in GMT +8 as a tuple of (date, time)"""
    internet_time = requests.get("http://just-the-time.appspot.com/").text
    date_time = datetime.datetime.strptime(internet_time.strip(), "%Y-%m-%d %H:%M:%S")
    date_time += datetime.timedelta(hours = 8, minutes = 0)

    date = date_time.strftime("%Y-%m-%d")
    time = date_time.strftime("%H:%M:%S")

    return date, time

def internet(host="8.8.8.8", port=53, timeout=3):
    """
       Checks for internet.

       Host: 8.8.8.8 (google-public-dns-a.google.com)
       OpenPort: 53/tcp
       Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception as ex:
        print(ex)

    return False

################################################################################
# Setup
################################################################################

# Start timer
start_time = time.time()

# Init Pyrebase app and database
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
db = firebase.database()

print("FIREBASE INITIALISED")

not_first_run = False

################################################################################
# Main loop
################################################################################

with open(LOG_FILE_NAME + ".csv", "a+", encoding = "utf-8", newline = "") as f:

    # Create writer object
    writer = csv.writer(f, delimiter = ",")

    # Move cursor to beginning
    f.seek(0, 0)

    # If the file is empty, initialise it
    if len(f.read()) == 0:
        print("Empty file! Initialising!\n")
        writer.writerow(["date", "time", "location/device", "people", "image", "location/device", "people", "image", "..."])

    # Begin main loop
    while True:
        # Re-init appending sub-lists for this iteration
        csv_append_list = []

        # Internet connectivity check
        # INFINITE LOOP IF THERE'S NO INTERNET
        while not internet():
            print("NO INTERNET. RECONNECTING...")
            print("Current total runtime:", time.time() - start_time)

            time.sleep(5)

        # Get current date-time tuple and write it to Firebase
        csv_append_list.extend(get_online_gmt_8_time())

        db.child("logger_time").update({"date": get_online_gmt_8_time()[0], "time": get_online_gmt_8_time()[1]})

        # Reset the database fetch dict
        val_dict = None

        # For as long as the dictionary is empty, keep checking!!
        while not val_dict:
            # Fetch the real-time database as an ordered dict
            try:
                val_dict = db.get().val()
            except:
                # Check for internet otherwise
                while not internet():
                    print("NO INTERNET. RECONNECTING...")
                    print("Current total runtime:", time.time() - start_time)

                    time.sleep(5)

            # If it exists, populate the appending sublist
            if val_dict:
                for i, j in val_dict.items():
                    # If the dictionary object isn't a logger_time object, then it is a device object
                    if i == "logger_time":
                        continue

                    ## Append device name/location ##
                    csv_append_list.append(i)


                    ## Append number of people ##
                    if type(j.get("people")) != str:
                        csv_append_list.append(j.get("people"))
                        db.child(i).update({"people": "WAITING"})
                    else:
                        # Reflect on the database if the next time the logger checks, we're still waiting for data
                        csv_append_list.append(j.get("people"))
                        db.child(i).update({"people": "FAILED"})


                    ## Append corresponding name of image ##
                    if j.get("last_image_captured").get("image_status") == "available":
                        image_file_name = j.get("last_image_captured").get("image_file_name")

                        db.child(i).child("last_image_captured").update({"image_status": "pulled", "image_file_name": ""})
                        csv_append_list.append(image_file_name)

                   # And reflect on the database if no new images were logged since the last time the logger checked
                    else:
                        image_file_name = j.get("last_image_captured").get("image_file_name")

                        db.child(i).child("last_image_captured").update({"image_status": "waiting", "image_file_name": ""})
                        csv_append_list.append(image_file_name)


        # If this is not the first run, write the sublist
        if not_first_run:
            writer.writerow(csv_append_list)

            # Force write from program buffer to OS buffer
            f.flush()

            # Force write from OS buffer to file
            os.fsync(f)

            print("LOGGED:", csv_append_list)
        else:
            not_first_run = True
            print("FLUSHING OLD DATA:", csv_append_list)


        for i in range(LOG_FREQUENCY):
            print(".", end="")
            time.sleep(10)

        print("\nCurrent total runtime:", time.time() - start_time)
