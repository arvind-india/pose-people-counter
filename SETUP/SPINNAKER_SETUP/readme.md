This is for using the Blackfly S series cameras with the people counter!    
Author: [methylDragon](https://github.com/methyldragon)



### Instructions

1. ENSURE that you're installing on an **Ubuntu 18.04** machine ONLY, and that your setup is using the same setup that's outlined in the setup docs in the root of this people counting package.
2. Ensure that you've **set up your Python ML** environment using the scripts in setup-scripts
3. Untar the tarball, and follow the README inside. **DO NOT wing it** and run the shell script alone. **There are other dependencies to install!**
4. Copy paste everything inside the Python folder into `~/miniconda3/envs/ML/lib/python3.5`



### To test

Run the `TEST_SCRIPT.py` script! It should save an image if it works! Detaching the camera might be a problem, but if it save a couple of images you shouldn't need to worry.



### Notes

You will not be able to find the explictly stated correct package versions and packages on the official Point Grey site. Some hacky stuff were done.

(I basically installed the SDK for Ubuntu 18.04 and got the files for the Python 3.6 library, but renamed it to make it usable in Python 3.5)

It works! That's good enough!