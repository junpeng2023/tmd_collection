

# Python

## 1. os file system

``````

1. for <> in os.listdir(<>)


2. os.path.exists(video_path)
 #-- to use a bool to see whether the path exists or not

3. if os.path.isdir(<>)
 #-- to use a bool to classify the folder_path and file_path, when we want to use folder_path, otherwise not able for iteration
O1 
# Path
path = '/home/User/Documents/file.txt'
  
# Check whether the 
# specified path is an
# existing directory or not
isdir = os.path.isdir(path)
print(isdir)
  
  
# Path
path = '/home/User/Documents/'
  
# Check whether the 
# specified path is an
# existing directory or not
isdir = os.path.isdir(path)
print(isdir)


4. os.getcwd()
# Print current working directory

5. 

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
           
1

``````

### 3.2 Filesystem and display

``````

1. 
<>=cv2.imread(<path_to_image_file>)
O1 cv.imread(imgpath)
Bsp1. 
imglist = png
imglist.sort(key=lambda x:int(x[6:-4]))

imgpath=os.path.join(file,imglist[0])


2.
cv2.imshow('<title_for_display>',<image_file>)

3.
cv2.destroyAllWindows()
#--


4.
cv2.waitKey()
#-- to adjust the play speed when running the script and enable key operations
O1 cv2.waitKey()&0xFF
O1 cv2.waitKey(0)&0xFF
O2 cv2.waitKey(100)&0xFF



5. 
<video_in_code>=cv2.VideoCapture(<video_path>)
#-- to read a video from a path, otherwise the video is not in the code
O1
video = cv2.VideoCapture(video_path)
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




``````

### 3.4 Draw

``````

1.
cv2.rectangle()




``````