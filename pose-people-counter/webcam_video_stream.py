# ADAPTED FROM: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

# Class for Threading video streams in OpenCV
# Heavily edited by methylDragon for specific use with the pose_people_counter

from threading import Thread, Lock
import numpy as np
import cv2
import time
import os

try:
    import PySpin
except:
    pass

def blend_background(background, overlay):
    """Overlay foreground image onto background, remove all black pixels. (Alpha supported!)"""
    # Split out the transparency mask from the colour info
    size = overlay.shape[2]

    overlay_img = overlay[:,:,:3] # Grab the BGR planes

    if size == 4:
        overlay_mask = overlay[:,:,3:]  # And the alpha plane
    else:
        # Compute element wise maximum for the BGR planes
        overlay_mask = np.maximum(overlay[:, :, 0], overlay[:, :, 1])
        overlay_mask = np.maximum(overlay_mask, overlay[:, :, 2])

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    background_output = (background * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_output = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(background_output, 255.0, overlay_output, 255.0, 0.0))

def rescale(frame, percent=50):
    """Rescale image to target stated percentage."""
    if percent == 100:
        return frame

    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)

    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

class WebcamVideoStream:
    def __init__(self, src=0, USING_BLACKFLY_S=False, FPS=24,
                 CAMERA_W=1280, CAMERA_H=720,
                 SCALING_PERCENTAGE=100,
                 DISPLAY_VIDEO=True, DISPLAY_W=640, DISPLAY_H=480,
                 WINDOW_NAME="Display", RECORD_VIDEO=False,
                 VIDEO_OUTPUT_FILENAME="output.avi"):

        # Set the BLACKFLY S use parameter
        self.using_blackfly = USING_BLACKFLY_S

        # Initialise the video stream
        if self.using_blackfly:
            # Acquire and init the camera
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            self.camera = self.cam_list.GetByIndex(0)
            self.camera.Init()

            # Configure the camera node
            self.camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.camera.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)

            # Initialise the capture
            self.camera.BeginAcquisition()
        else:
            # Set video stream object (select webcam)
            self.stream = cv2.VideoCapture(src)

            # Configure parameters
            self.stream.set(3, CAMERA_W); # Stream width
            self.stream.set(4, CAMERA_H); # Stream height

        ## Initialise variables ##

        # Frame scaling
        self.fps = FPS
        self.scaling_percentage = SCALING_PERCENTAGE

        if self.using_blackfly:
            self.width = CAMERA_W * self.scaling_percentage // 100
            self.height = CAMERA_H * self.scaling_percentage // 100
        else:
            self.width = int(self.get(3)) * self.scaling_percentage // 100  # float
            self.height = int(self.get(4)) * self.scaling_percentage // 100 # float

        # Display
        self.display_video = DISPLAY_VIDEO

        self.display_w = DISPLAY_W
        self.display_h = DISPLAY_H

        self.window_name = WINDOW_NAME

        # Recording
        self.record_video = RECORD_VIDEO
        self.video_output_filename = VIDEO_OUTPUT_FILENAME

        # Frame initialisations
        # Sub-frame to ensure the displayed frame never gets sent unscaled (for threads)
        self.sub_frame = None
        self.overlay_frame = np.zeros((self.height, self.width, 3), np.uint8)

        # Variables used to indicate if the thread should be stopped
        self.displaying = False
        self.stopped = False

        # Initialise thread lock object
        self.lock = Lock()

        # Read and process the first frame from the stream
        if self.using_blackfly:
            # Acquire the latest image off the camera buffer
            try:
                self.blackfly_image = self.camera.GetNextImage()

                if self.blackfly_image.IsIncomplete():
                    self.ret = "False"
                else:
                    # Dereference the acquired pointer object
                    self.sub_frame = self.blackfly_image.Convert(PySpin.PixelFormat_BGR8).GetNDArray()
                    self.ret = "Unknown"
            except:
                self.ret = "False"

            self.frame = rescale(self.sub_frame, percent = self.scaling_percentage)

        else:
            self.ret, self.frame = self.stream.read()

        # Initialise video recording parameters
        if self.display_video or self.record_video:
            if self.record_video:
                print("VIDEO OUTPUT FILE:", self.video_output_filename)

                # Initialise videowriter
                self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter(self.video_output_filename, self.fourcc, 24.0,
                                      (int(self.width), int(self.height)))

            # Start display (and record) thread
            self.start_show(self.window_name)

    def start(self):
        """Start the thread to read frames from the video stream."""
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        "Keep looping infinitely until the thread is stopped."""
        while True:
            # If the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            with self.lock:
                # Otherwise, read the next frame from the stream
                if self.using_blackfly:
                    # Acquire the latest image off the camera buffer
                    try:
                        self.blackfly_image = self.camera.GetNextImage()

                        if self.blackfly_image.IsIncomplete():
                            self.ret = "False"
                        else:
                            # Dereference the acquired pointer object
                            self.sub_frame = self.blackfly_image.Convert(PySpin.PixelFormat_BGR8).GetNDArray()
                            self.ret = "Unknown"
                    except:
                        self.ret = "False"

                else:
                    # Read from the input video stream
                    self.ret, self.sub_frame = self.stream.read()

            self.frame = rescale(self.sub_frame, percent = self.scaling_percentage)

            # Introduce a delay set by the FPS parameter
            time.sleep(1 / self.fps)

    def read(self):
        """Return the frame most recently read."""
        return self.ret, self.frame

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True

    def get(self, property):
        """Get VideoCapture object property"""
        return self.stream.get(property)

    def start_show(self, window_name):
        """Initialise display (and record) thread."""
        if self.displaying == False:
            Thread(target=self.show, args=([window_name])).start()
            self.displaying = True

    def show(self, window_name):
        """Thread to display the current frame."""

        # Create a resizeable window to display the display in
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_w, self.display_h)

        while True:

            start_time = time.time()
            # If the thread stop flag is triggered, kill the thread
            if self.stopped:
                with self.lock:
                    self.displaying = False

                    if self.using_blackfly:
                        # Cleanup the blackfly related variables and objects
                        self.camera.EndAcquisition()
                        self.camera.DeInit()

                        del self.camera

                        self.cam_list.Clear()
                        self.system.ReleaseInstance()
                    else:
                        # Close the input video stream
                        self.stream.release()

                    if self.record_video:
                        # Close the output video stream
                        self.out.release()
                return

            with self.lock:
                # Show the read and processed frames (with possible recording!)
                self.display_frame = blend_background(self.frame, self.overlay_frame)
                cv2.imshow(str(window_name), self.display_frame)

                # If video recording is enabled, output to the video file
                if self.record_video:
                    self.out.write(self.display_frame)

                # Introduce a delay set by the FPS parameter
                time.sleep(1 / self.fps)

            # Required by OpenCV to run. Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
