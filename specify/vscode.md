

# Python

## 1. os file system

``````

1. for <> in os.listdir(<>)


2. 



``````


## 2. Basics

### 2.1 for loop

``````
1.

for <> in range(<>):


2. 
for <> in <>:

3. 
for <> in enumerate(<>):

4.
for <> in os.listdir():

5. 
for 





``````


### 2.2 if ... else ...

``````

1.
if <file>.endswith('<>'):
#-- to only select those files with certain formats
O1 if filename.endswith('.jpg'):
O2 if filename.endswith('.png'):

2.




``````

## 3. OpenCV

### 3.1 Color
``````

1.
cv2.cvtColor(<single_image>,<Color Space Conversions>)
#--
// <Color Space Conversions> O1 cv2.COLOR_BGR2GRAY
e.g.1. #转变为HSV颜色空间
img_hsv=cv2.cvtColor(img_original,cv2.COLOR_BGR2HSV)

e.g.2. 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
1

``````

### 3.2 Filesystem and display

``````

1. 
cv2.imread(<path_to_image_file>)
#-- to read an image located in the filesystem into the code
O1 cv.imread(imgpath)
Bsp1. 
imglist = png
imglist.sort(key=lambda x:int(x[6:-4]))

imgpath=os.path.join(file,imglist[0])


2.
cv2.imshow('<title_for_display>',<image_file>)
#-- to show the image or video in a window with title

3.
cv2.destroyAllWindows()
#-- to remove the window which is not used any more after imshow()


4.
cv2.waitKey()
#-- to adjust the play speed when running the script and enable key operations
O1 cv2.waitKey(0)&0xFF
O2 cv2.waitKey(100)&0xFF

5.
cv2.imwrite()
#-- to write a image in the code into the filesystem

``````

### 3.3 camera_calibration

``````
1.
ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE)

2.
image = cv2.drawChessboardCorners(image, 
                                            CHECKERBOARD, 
                                            corners2, ret)
#-- to draw the corners on the chessboard image file, otherwise no feature for calibration

3. 
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

4.
corners= cv2.cornerSubPix()
O1 corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
#-- 在原角点的基础上寻找亚像素角点

5. newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
#-- 
O1 newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))



``````

### 3.4 Draw

``````

1.
cv2.rectangle()
#-- to draw bounding boxes or something equivalent

2.
cv2.circle()
#-- to draw a point e.g. two sides of a lane

3.





``````



### 3.5 Display Window Operations

``````
1. cv2.namedWindow(<window_title>,cv2.WINDOW_NORMAL)
O1
cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)

2. cv2.resizeWindow('<window_title>'.<length>,<height>)
O2      
cv2.resizeWindow('findCorners', 640, 480)



``````

### 3.6 detection

```
1.
cv2.Canny()
#-- to do the edge detection with the output of a white/black one 
e.g.1. edges = cv2.Canny(blurred, 50, 150)
e.g.2. 



2.
cv2.fillPoly()


3.
cv2.



```

### 3.7 shape changing


```

1.
cv2.dilate()
#-- to jiacu a line or word

2.
cv2.erode()
#-- to bianxi a line or word




```

### 3.8 Noise reduction

```

# Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)



```

### 3-A Troubleshooting


#### 1
```


/home/ziwei/anaconda3/bin/python "/media/ziwei/PortableSSD/Junpeng/duckietown_cv/csdn_ calibration.py"
[ WARN:0@0.006] global loadsave.cpp:244 findDecoder imread_('calibration_1.mkvframe0257.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/duckietown_cv/csdn_ calibration.py", line 40, in <module>
    print('img.shape',img.shape)
AttributeError: 'NoneType' object has no attribute 'shape'

```

##### Solution
  e.g. img_path = os.path.join('/home/ziwei/Desktop/calibration_3', filename)
  --- as file_name is e.g. frame0001.jpg and the real directory is e.g /home/ziwei/Desktop/calibration_3, so we have to get them together to be the input for "cv2.imread(<>)"


#### 2

```

/home/ziwei/anaconda3/bin/python /media/ziwei/PortableSSD/Junpeng/duckietown_cv/camera_cali.py
[ WARN:0@0.006] global loadsave.cpp:244 findDecoder imread_('calibration_1.mkvframe0257.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/duckietown_cv/camera_cali.py", line 65, in <module>
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.error: OpenCV(4.7.0) /io/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

```

##### solution
  e.g. img_path = os.path.join('/home/ziwei/Desktop/calibration_3', filename)
  --- as file_name is e.g. frame0001.jpg and the real directory is e.g /home/ziwei/Desktop/calibration_3, so we have to get them together to be the input for "cv2.imread(<>)"

#### 3

```

[ WARN:0@1.989] global cap_v4l.cpp:982 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
[ERROR:0@1.990] global obsensor_uvc_stream_channel.cpp:156 getStreamChannelGroup Camera index out of range
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/duckietown_cv/csdn_ calibration.py", line 88, in <module>
    h1, w1 = frame.shape[:2]
AttributeError: 'NoneType' object has no attribute 'shape'

```

##### solution


#### 4

```
from common import splitfn
# the package common is missing

```

##### solution
 --- get the file common.py from the git of opencv/sample/python/ to the same directory as the code




#### 5

```
--- always the same image written to the undistorted image folder

```

##### possible reasons
--- e.g. dst = cv2.undistort(image, matrix, distortion, None, newcameramtx), while image is not from imread but from another place such as glob


##### solution
--- should be img instead of image


#### 6

```
--- always get just a few images instead of those in the whole directory

```

##### solution
 --- to use less images selected in a smaller range instead of using frames of a whole video
 --- use the cv2.destroyWindow, otherwise the display imshow would not cease


## 4. CUDA

### 4-A Troubleshooting
 
 #### 1

 
 ```
 
    RuntimeError: CUDA out of memory.
    RuntimeError: CUDA error: device-side assert triggered.
    RuntimeError: CUDA error: launch failed.

 
 ```

 ##### solution

 ```
  -- use the DDP to speed up the training process
  
 
 ```


 ## 5. numpy

 ### 5.1

 ```
 1.
 <>=np.array([])
 e.g.1. 
 lower_yellow = np.array([20, 100, 100])
 upper_yellow = np.array([40, 255, 255])
 
 2.

 e.g.1
 mask = np.zeros_like(img)

 
 ```