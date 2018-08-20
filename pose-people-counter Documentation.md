# pose-people-counter Documentation

Implementor: methylDragon

An accurate, tunable people counter that uses pose estimation and can be deployed in a distributed network that pushes to the cloud! Records images, videos, and logs to a .csv!

Includes heavily edited code, based off of https://github.com/eldar/pose-tensorflow

It also uses the model wholesale from that repo

----

## Table Of Contents <a name="top"></a>

1. [Setup](#1)  
   1.1   [Purpose](#1.1)    
   1.2   [Implementation](#1.2)    
   1.3   [Scripts](#1.3)    
   1.4   [Challenges](#1.4)    
2. [Working Theory](#2)    
   2.1   [IoT](#2.1)    
   2.2   [Software](#2.2)    
   2.3   [Model](#2.3)    
3. [Configurable Parameters](#3)  
4. [Thanks](#4)  
5. [References](#5)  



## 1. Setup <a name="1"></a>

### 1.1 Purpose <a name="1.1"></a>

[go to top](#top)

For use in any situation that requires counting the number of people in a location over time. (No tracking is done, the script only counts the number of people in a snapshot.)



### 1.2 Implementation <a name="1.2"></a>

[go to top](#top)

Deploy OpenCV + Tensorflow on a Linux computer (like the [ASUS VivoMini](https://www.asus.com/sg/Mini-PCs/VivoMini-VC66/)).

Run the [model](https://github.com/eldar/pose-tensorflow), but using my edited code that is properly threaded and included with API calls to Firebase.

View the real-time data on the 'dashboard' that is the Firebase cloud database.

Run the logger Python script to log the real-time data to a .csv for visualisation and analysis.



### 1.3 Scripts <a name="1.3"></a>

[go to top](#top)

I wrote a couple of scripts to help with setup. And wrote and edited a couple of other scripts to get this to work.

Please use them! For the full tutorial, please check out the setup tutorial.



#### **Setup**

For more info about the setup, check the **Setup Documentation** markdown file!

Find these in the SETUP/setup_scripts folder. Run them in order!:

- **1_convenience_tools_install**
  - Installs some handy convenience tools
- **2_conda_ml_setup**
  - Installs conda, and sets up a machine learning environment (ML)
- **3_people_counter_python_dependencies_install**
  - Installs the dependencies for people counting (and Tensorflow!
- **4_opencv_setup**
  - Compiles and installs OpenCV, set up specifically for the environment, within the environment

Run them like so:

```bash
$ cd <setup_scripts_directory>

# Enable execution
$ chmod +x ./*

# Remember to read the output and pay attention! User input is REQUIRED
$ ./1_convenience_tools_install
$ ./2_conda_ml_setup
$ ./3_people_counter_python_dependencies_install
$ ./4_opencv_setup
```



Also remember to **compile the pose-tensorflow scripts** after downloading the models!

```bash
$ cd pose-people-counter
$ ./models/coco/download_models.sh # You might have to chmod +x this
$ chmod +x compile.sh
$ ./compile.sh
```



#### **Runtime**

They're in the pose-people-counter folder

Run these (remember to chmod +x them of course!):

You will need multiple terminals

```bash
$ python pose_people_counter.py
$ python pose_people_counter_logger.py
```



You may then create a database on Firebase and input the credentials. You really only need the database URL, and maybe an API key if you set the permissions to be very open. Tightening it up will require more configurations that you should be able to find guides on setting up online.



### 1.4 Challenges <a name="1.4"></a>

[go to top](#top)

Raspberry Pis (RPis) can't run the model for some reason. The isssue has to do with this model specifically. Tensorflow and OpenCV will be able to run, but on using the model, an unknown **Bus error** is triggered with not much explanation.

Debugging using GDB doesn't do much. But I suspect that the processing time on an RPi and the memory usage will be prohibitive to deploy this model on RPis anyway.



## 2. Working Theory <a name="2"></a>

### 2.1 IoT <a name="2.1"></a>

[go to top](#top)

Implementing the system with the principle of edge computation was a good goal to have. That way we minimise the data transferred over the air, and avoid having a single point of computation.

The choice of using a NoSQL Firebase realtime-database (aside from being easy to set up) was also partially motivated by the ability to double it up as a 'dashboard' of sorts. Additionally, it is easy to set up a dashboard by polling the Firebase database for the real-time data.

When the logger logs data, it also pulls the data away from the Firebase and puts the 'people' variable for each entry it has pulled into a 'WAITING' state. If the 'WAITING' state is still there the next time Firebase is polled by the logger, it is put in a 'FAILED' state. This allows you to very easily keep track of the status of your entire system by just looking at the database.

Additionally, the last update time for the loggers as well as the individual devices are put up for more convenience. They're reported according to the local time on the devices (pulled from online.) So they should be synced if they poll at the same time, barring any weird exceptions.



### 2.2 Software <a name="2.2"></a>

[go to top](#top)

Some features were implemented in order to improve the performance or aid in tuning.

- Threading was implemented for the video streams to allow the video display to not freeze when inferences are occuring. A custom video stream class was written for this purpose. The annotations from the previous inference persists until replaced by a succeeding one.
- Individual clusters of pose keypoints that are associated with a distinct person are coloured differently to aid in intuitively understanding the outputs of the model.
- Many exposed user parameters are available near the top of the script to allow you to change the display frame size, display frame name, toggle recording, toggle display, set frame-rate, etc.

The inferencer is setup to infer every 2 minutes, while the logger logs every 5 minutes.

Image capturing is configured to run every 30 minutes.



### 2.3 Model <a name="2.3"></a>

[go to top](#top)

I did not train or create a model on my own. I instead used one by Eldar et al. from the open source implementation on [Github](https://github.com/eldar/pose-tensorflow).

Paper: https://arxiv.org/pdf/1612.01465.pdf    
Conference: https://www.youtube.com/watch?v=kdV2sdZ9TWg

The model uses resnet-101 to detect body parts. Then it extracts features (sometimes from a hidden layer) and applies math to derive clusters of pose keypoints associated with individuals! Notably, it uses both **spatial** and **temporal** relationships. In other words, it keeps track of the relative distances of the pose keypoints on the picture frame in a single instance, as well as **across frames!**

It then counts the number of people detected and uses that.

A rough outline of the model goes as such:

(If I read the paper correctly.)

> Human pose layouts are defined as a sparse graph of 17 keypoints. The heads have a higher density of keypoints.

- Find heads using necks and tops of heads. Then find chins. Chins are never linked to other chins when proposing a pose configuration. The chin will be the 'root' of pose propositions, features that fall outside of the propositions are penalised when a chin is detected.
  - Heads and chins are not necessary for people detection, but they increase the accuracy and reliability of the detection. The network will still attempt to find a pose.
- Iterate through the heads. When a head is selected, the person it belongs to is a 'selected person'
- Spatially propagate from each feature to detect shoulders, elbows, arms, and then legs, and so on.
- Generate a pair-wise prediction of pose keypoints for spatial and temporal frames, independently
  - This is configurable on pose_cfg_multi.yaml, but I did not configure it.
- Group the poses using attractive/repulsive edges that have costs inversely proportional to distance.
- Feed the groups through a layer that's focused on joint prediction, with a loss tagged the fact that the pose keypoints correspond to the body joints of an individual.



## 3. Configurable Parameters <a name="3"></a>

#### **Debug**

- **DEBUG** (*bool*): Verbose mode
- **DEBUG_INFERENCES** (*bool*): Set inference frequency to super frequent, and disable pushing to Firebase

#### **Recording**

- **RECORD_VIDEO** (*bool*): Record the annotated video
- **VIDEO_OUTPUT_FILENAME** (*str*): Set output video file name
- **RECORD_IMAGES** (*bool*): Enable image recording
- **IMAGE_CAPTURE_FREQUENCY** (*int*): Enable annotated image capture (in seconds)
- **IMAGE_OUTPUT_FOLDER** (*str*): Set output folder for recording images in normal mode
- **DEBUG_IMAGE_OUTPUT_FOLDER** (*str*): Set folder for recording images in debug_inference mode

#### **Display**

- **DISPLAY_VIDEO** (*bool*): Display the video!
- **DISPLAY_W** (*int*): Set display width
- **DISPLAY_H** (*int*): Set display height
- **WINDOW_NAME** (*str*): Set display window name

#### **Processing**

- **FPS** (*int*): Set framerate (how fast frames are processed, NOT DISPLAYED or BUFFERED!)
- **POINT_THRESHOLD** (*int*): How many pose keypoints a pose needs in order to be counted as a person
- **USING_BLACKFLY_S** (*bool*): Set to True if you're using a Blackfly S camera
- **CAMERA_NUMBER** (*int*): OpenCV video stream camera number (0 should work usually)
- **CAMERA_W** (*int*): Camera input stream width
- **CAMERA_H** (*int*): Camera input stream height
- **USE_CANNED_VIDEO** (*bool*): Set to True if using a pre-recorded video as input
- **CANNED_VIDEO_PATH** (*str*): The path to the canned video
- **INFERENCE_FREQUENCY** (*int*): The frequency of inferences (in seconds)
- **SCALING_PERCENTAGE** (*int*): How much to scale the image (in percentage) (Affects sensitivity and false positives)

#### **Firebase**

- **PUSH_TO_FIREBASE** (*bool*): Enable pushing to Firebase
- **DEVICE_NAME** (str): The name of the device/location. The data pushed to Firebase will be grouped under this name.

(And finally...)

- **FIREBASE_CONFIG** = {
  "apiKey": "API-KEY-HERE",
  "authDomain": "",
  "databaseURL": "DATABASE-URL-FOR-FIREBASE-HERE (???.firebaseio.com)",
  "storageBucket" : ""
  }



## 4. Thanks <a name="4"></a>

Bi Qing for helping to decide what model to use!

Ebi for the equipment and advice

Eldar et al. for the very robust people tracking implementation



## 5. References <a name="5"></a>

```
@inproceedings{insafutdinov2017cvpr,
    title = {ArtTrack: Articulated Multi-person Tracking in the Wild},
    booktitle = {CVPR'17},
    url = {http://arxiv.org/abs/1612.01465},
    author = {Eldar Insafutdinov and Mykhaylo Andriluka and Leonid Pishchulin and Siyu Tang and Evgeny Levinkov and Bjoern Andres and Bernt Schiele}
}

@article{insafutdinov2016eccv,
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    booktitle = {ECCV'16},
    url = {http://arxiv.org/abs/1605.03170},
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele}
}
```






```
                            .     .
                         .  |\-^-/|  .    
                        /| } O.=.O { |\     
```

​        

------

 [![Yeah! Buy the DRAGON a COFFEE!](.assets/COFFEE%20BUTTON%20%E3%83%BE(%C2%B0%E2%88%87%C2%B0%5E).png)](https://www.buymeacoffee.com/methylDragon)