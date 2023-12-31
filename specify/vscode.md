

# Python

## 1. os file system

``````

1. for <> in os.listdir(<>)


2. with open('<file_name>', '<mode>') as f:
O1.
e.g.1.
with open('car_line.txt', 'r') as f:
O2.
e.g.1.
with open('car_line.txt', 'w') as f:
O3.
e.g.1.



3.


``````

### 1.1 os.path

```

1.
<a two dimentional list which contains head and tail> =os.path.split(<path_to_file>)
e.g.1.
head_tail = os.path.split(path)

2.


```


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
O3 cv2.waitKey(1)

5.
cv2.imwrite()
#-- to write a image in the code into the filesystem

6.
cv2.namedWindow('<title_for_display>',<type>)
#-- to display the images in a window, whose size can be changed unlike the cv2.imshow()
e.g.1.
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

V2:<type>
O1.cv2.WINDOW_NORMAL
#-- to randomly set the size of the window
/--weblink: O1 https://blog.csdn.net/fanjiule/article/details/81606596
            #-- a csdn website to show how to use the cv2.namedWindow

O2.



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


### 3.9 optical flow

```
cv2.calcOpticalFlowPyrLK()
#-- 该函数用于计算稀疏光流。它接受前一帧图像和当前帧图像作为输入，并根据给定的特征点或兴趣区域跟踪这些特征点在两个图像之间的位置变化。函数返回被成功追踪的特征点的新位置以及一个状态值。

cv2.calcOpticalFlowFarneback()：
#--该函数用于计算稠密光流。它接受前一帧图像和当前帧图像作为输入，并估计整个图像中每个像素点的运动向量。函数返回每个像素点的光流向量值。

cv2.goodFeaturesToTrack()：
#--该函数用于在图像中检测良好的特征点。它接受输入图像和一些参数，如角点检测方法、特征点数量等，并返回检测到的良好特征点的坐标。

cv2.drawOpticalFlow()：
#--该函数用于可视化光流结果。它接受一张彩色图像和光流向量作为输入，并在图像上绘制箭头表示运动方向和强度。
————————————————
版权声明：本文为CSDN博主「陈子迩」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_45303602/article/details/133814463

5.
cv2.goodFeatureToTrack()
#-- to determine which points should be tracked



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

### 3.9 multiple images

```
1.
cv2.addWeighted()
#-- get two images together
e.g.1 
complete = cv2.addWeighted(initial_img, alpha, line_image, beta, lambda)


2.
cv2.bitwise_and()

3.
cv2.bitwise_or()

```

### 3.10 ROI operations

```
1.
e.g.1.
roi = cv2.selectROI("Select ROI", frame, False, False)



```


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


 ## 5. datatypes

 ### 5.1 numpy

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

 ### 5.2 list/dict

 ```
 1. eval()
  e.g.1.
  line = eval(line) 

 2. <list_name>.strip('<element to be removed>')
 e.g.1.
  line = line.strip('\n')


 3. <list_name>.item()
 /--weblinks: https://www.programiz.com/python-programming/methods/dictionary/items
  e.g.1.
   bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())

 
 
 
 ```


 ### 5.3 arrays

 ```
 1.
 <>.ravel()
#-- to flatten a multi-dimensional array into a 1-dimensional array
 e.g.1.
 a,b=new.ravel()
 c,d= old.ravel()
 



```


 ## 6. 