# pose-people-counter Setup Tutorial

Author: methylDragon  
Setting up the pose people counter!     

------

## Pre-Requisites

**Hardware**

- A 64-bit computer!
- Some camera that can communicate with the PC

**Software**

- **Ubuntu 18.04 ONLY**
- The setup pack that comes with this tutorial
- The scripts that come with this tutorial
  - Python, OpenCV and Tensorflow knowledge will come in handy!
  - **Don't know Python?** Here's a handy [tutorial repo](https://github.com/methylDragon/coding-notes)!
- Basic Linux knowledge
  - Go play some [Terminus](http://web.mit.edu/mprat/Public/web/Terminus/Web/main.html)!



## Table Of Contents <a name="top"></a>

1. [Introduction](#1)  
   1.1   [Purpose](#1.1)    
   1.2   [Implementation](#1.2)    
   1.3   [Scripts](#1.3)    
   1.4   [Challenges](#1.4)    
2. [Setup](#2)    
   2.1   [Overview](#2.1)    
   2.2   [System Setup](#2.2)    
   2.3   [Tools and Envs](#2.3)    
   2.4   [Install Dependencies](#2.4)    
   2.5   [Compile Compilables (And download the model!)](#2.5)    
   2.6   [Validate](#2.6)    
3. [Using the Blackfly S Camera](#3)  





## 1. Introduction <a name="1"></a>

We're going to implement people counting. This is implemented in a way that will allow you to deploy it across various computers, forming a distibuted network of sensors that push data to the cloud. (If you intend to do this on more than one computer, be sure to change the name of the devices!!)

We're going to use OpenCV and Tensorflow to do this, and push the number of people counted to Firebase! The Firebase database is realtime and can be used as a pseudo "dashboard."

Then we'll pair this with a logger program that logs the data to .csv files so we can plot the time-series data.



## 2. Setup <a name="2"></a>

### 2.1 Overview <a name="2.1"></a>

[go to top](#top)

We're going to want to get our System set up first! 



#### **Setup Steps**

**System Setup**

- [ ] **Make sure you have a computer with a decent amount of RAM and hard drive space.** (8 GB RAM should be safe)
- [ ] Get **Ubuntu 18.04 on it**

**Tools and Envs**

- [ ] Install methyl-convenience-tools
- [ ] Install Conda
- [ ] Create the conda environment for the people-counting

**Depnedencies**

- [ ] Install Python dependencies
- [ ] Install Tensorflow

**Compile Compilables**

- [ ] Install OpenCV
- [ ] Download the models
- [ ] Compile people-counter libraries

**Validate**

- [ ] Place the camera at good vantage points!
- [ ] Run the code!
- [ ] Review the performance of the model!
- [ ] Check the Firebase!
- [ ] Check the .csv files logged!



### 2.2 System Setup <a name="2.2"></a>

[go to top](#top)

> **IMPORTANT NOTE:**
>
> This model doesn't seem to run on the RPi (after compiling and setting up everything, running the model results in an unexplained Bus error. No solution was found yet.) so an ASUS VivoMini PC was substituted. This is why I said to use a 64-bit system.
>
> Just make sure you install **Ubuntu 18.04** and have a monitor, keyboard, and mouse ready.



#### **1. Ready your System**

Yeah! Make sure haha...



#### **2. Get Ubuntu 18.04**

You need this mainly for Blackfly compatibility, but it's always good to take the latest Ubuntu LTS Distribution anyways. And with this we're safely within communities and support for the dependencies.

https://www.ubuntu.com/download/desktop

https://linoxide.com/distros/install-ubuntu-18-04-dual-boot-windows-10/



### 2.3 Tools and Envs <a name="2.3"></a>

[go to top](#top)

#### **Preamble**

Run this to **enable the execution of all install cheatscripts!**

This also assumes basic understanding of Linux commands. If you don't have this understanding, kindly go play some [Terminus](http://web.mit.edu/mprat/Public/web/Terminus/Web/main.html)!

If not, **Ctrl-Alt-T** will open up Terminal for you!

```bash
$ chmod -R +x . # You might have to sudo this
```

The scripts are in SETUP/setup_scripts



#### **1. Install methyl-convenience-tools**

Run the following in the **setup_scripts folder**

```bash
# Install!
$ ./1_convenience_tools_install # Remember to read!
```

> You may also feel free to peruse the rest of my [quick-install-scripts](https://github.com/methylDragon/quick-install-scripts/)!
>
> \- CH3EERS!



#### **2. Install Conda**

Run the following. Remember to read also!

```bash
# Install! ALSO CHOOSE THE RIGHT ONE

$ ./2_conda_ml_setup # Remember to read!
```

You might have to run this multiple times if conda fails the first few times around.

If it does **please close and reopen terminal.**



#### **3. Create a conda environment for the people-counting**

Follow step 2! Just continue reading! Everything gets set up for you if you select the correct things :)

It'll create a conda environment called **ML.** Environments are especially useful here since OpenCV likes to mess up Python distributions if not properly contained!

If you ever mess anything up, locate your Conda environment folders, then go ahead and delete the corresponding folder for ML, and just start again from this section.



### 2.4 Install Dependencies <a name="2.4"></a>

[go to top](#top)

#### **1. Install Python dependencies**

> Psst. I'd recommend you follow this tutorial step-by-step. But if you're lazy, go ahead and run:
>
> Just **make sure that your pip is for python3!** If you don't know how, just do it step-by-step please!
>
> ```bash
> $ ./3_people_counter_python_dependencies_install
> ```

>  **Ensure you're in your ML environment!** All future steps will **assume** that you're in the ML environment!
>
> Don't skimp on this unless you want to risk messing up the system!

If you aren't, invoke the conda environment using:

```bash
$ source activate ML
```



Then:

```bash
$ pip install scipy scikit-image matplotlib pyyaml easydict cython munkres pyrebase
```

(We're using the Pyrebase API instead of the Firebase one because there actually isn't an officially supported distribution of the Firebase API for Linux for some strange reason.)

**Make sure everything installs! Pay attention!**



#### **2. Install Tensorflow**

> **Note:**    
> Since we've set up a conda environment, pip refers to pip3! If it isn't set up properly, then change all the commands below to pip3.

**Double check the version of your pip.**

It should say that it's for Python 3. If it is, you're good to go!

```bash
$ pip -V # If it isn't, check
$ pip3 -V

# If none of them are for Python 3, then
$ sudo apt install python3-pip
```



Time to install Tensorflow! (We'll be installing the CPU version. If you happen to have a GPU in your embedded computer for some reason, feel free to go ahead and use it! After configuring it of course...)

```bash
# Get the dependencies settled
$ sudo apt install libatlas-base-dev

# Andddddd... (MAKE SURE YOU'RE INSTALLING FOR PYTHON 3!)
$ pip install tensorflow

# WOW DONE. WOW. THIS USED TO TAKE AGES.
```



Ok! Validate your Tensorflow install! If this breaks, blame Google, cry a little, and then: https://www.tensorflow.org/install/

```python
# In your environment, fire up a Python 3 instance, and run this script

# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

If it outputs 'Hello, TensorFlow!'', congratulations! You're good to go!



### 2.5 Compile Compilables <a name="2.5"></a>

[go to top](#top)

#### **1. Install OpenCV**
> **THIS IS GOING TO TAKE AWHILE!** BE VERY CAREFUL!!

We're going to have to **build OpenCV from source**. In order to do this we need to make sure we have some stuff settled.

- Ensure that your Raspberry Pi **has a heatsink on it**
  - Otherwise, it **WILL overheat** and crash! (You might even burn it out!)
- Ensure that you **ALSO have a case with a fan on it!**
- Ensure you've **increased the swapspace on your Pi**
  - Otherwise it **WILL run out of RAM (at about 82%) and CRASH**

**Again, the build time will take anywhere from 3-26 hours!** (Average about 6~)



Aaaand, here we go!

```bash
# Install!
$ ./4_opencv_setup # Remember to read!!
```



#### **2. Download the models**

We need to download the pose estimation models (since they're too large to put on Github.)

Navigate to `pose-people-counter/models`, and pick the model you want to use. If we're going with the default, go into the `coco` folder.

Run

```bash
# Remember to read and enable execution with chmod +x
$ download_models.sh
```



#### **3. Compile people-counter libraries**

Run

```bash
./compile.sh
```



### 2.6 Validate <a name="2.6"></a>

[go to top](#top)

#### **1. Place the cameras at good vantage points!**

Higher is better! Try to ensure that no bags are detected as people, and ensure line of sight is maximised.



#### **2. Run the code!**

Go into the pose-people-counter folder and run the code!

Run the code! This includes the model code as well as the Firebase logger code.

```bash
# Remember to enable execution
$ chmod +x ./*

# Then run the scripts! (You will need multiple terminals)
$ python pose_people_counter
$ python people_counter_logger
```



#### **3. Tune and Review the performance of the model!**

Standard.

You'll want to tune the script's sensitivity by adjusting BOTH the **SCALING_PERCENTAGE** and **POINT_THRESHOLD** values.

Scaling percentage will affect the sensitivity of the model as well as the amount of false positives that occur.

The point threshold number is the number of points a pose needs to have in order to be considered a person, it's a useful tool for configuring the script to ignore false positive clusters.



#### **4. Check the Firebase!**

If you want to! If you configured it, that is.

In order to configure it, go into the pose_people_counter.py script, look at the top few lines under the `User configurable parameters` section, and plug in Firebase database credentials.



#### **5. Check the .csv files logged!**

Likewise! Make sure you're running the logger and have the logger configured to communicate with the Firebase database the pose_people_counter.py script is pushing to.



## 3. Using the Blackfly S Camera <a name="3"></a>

This is if you want to go crazy and splurge on a good small camera suited for these sorts of applications.

The Blackfly S camera requires that you install a couple of drivers. Luckily for you, I've already included the installation dependencies in a folder called SPINNAKER_SETUP. If you followed the instruction to use Ubuntu 18.04, this should be no sweat for you. If not, you probably should know enough to handle this on your own, especially if you forked out the money to get the camera, or know about it in the first place.

Just follow the instructions in the readme there.

Then configure the corresponding parameter on the `pose_people_counter` script. They're pretty self explanatory!





```
                            .     .
                         .  |\-^-/|  .    
                        /| } O.=.O { |\     
```

​        

------

 [![Yeah! Buy the DRAGON a COFFEE!](../.assets/COFFEE%20BUTTON%20%E3%83%BE(%C2%B0%E2%88%87%C2%B0%5E).png)](https://www.buymeacoffee.com/methylDragon)