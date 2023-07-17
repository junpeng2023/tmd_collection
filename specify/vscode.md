

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
           
1

``````

### 3.2 Filesystem and display

``````

1. 
cv2.imread(<path_to_image_file>)


2.
cv2.imshow('<title_for_display>',<image_file>)

3.
cv2.destroyAllWindows()
#--


4.
cv2.waitKey()
#-- to adjust the play speed when running the script and enable key operations
O1 cv2.waitKey(0)&0xFF
O2 cv2.waitKey(100)&0xFF
O
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




``````