# Flow of projects

# 0. How to collect materials

## weblinks:
```
1.
Youtube
#-- useful for animation e.g. in project realization

2.
github
#-- useful for code cloning


3.
CSDN
#-- useful for looking for all kinds of technical tutorials and troubleshooting as well e.g. in YOLO



```

# 1. Installation of something on Ubuntu

#### idea1. -- How to install something using anaconda

bug1: version incompatible?

# 2. CV_project

## 1. Duckietown_project

### 1. -- to get the pcl Pointcloud with given depth and rgb frames

#### idea1. -- How to install the pcl package?

/--weblink:
--O1 https://tony-laoshi.github.io/pcl%E5%BA%93%E5%9C%A8ubuntu20.04%E4%B8%8A%E7%9A%84%E5%AE%89%E8%A3%85/
#-- with how to install all dependencies

--S1. sudo apt install libpcl-dev

--S2. installing the python pcl
 O1 pip install python-pcl -vvvv
 bug1: errors with pcl common
  sol1: pip install python-pcl -vvvv
  sol2: 

--S3. to install the python3-pcl instead of python-pcl
 weblink: https://blog.csdn.net/zsssrs/article/details/120054425
 #-- how to get the package into anaconda
 --S3-1 use "sudo apt-get install python3-pcl"

--S4

--S5






#### idea2. -- to get the docker image for pcl and run on it

/--bug1: docker image can not be run


### 2. -- to get the rgb and depth frames from one given rosbag file

#### idea1. use the file rosbag2video.py 

``````
 /--weblink: https://github.com/mlaiacker/rosbag2video/tree/master
 #-- git repository for rosbag2video.py
 S1 use the terminal command such as "python3 rosbag2video.py --fps 25 --rate 1 -t /camera/aligned_depth_to_color/image_raw mpmp_working_2_2022-10-19-20-26-59.bag "
 /--bug1: can not use -s in the tmd, otherwise with error message
 "Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/rosbag/mp/mp_working/rosbag2video.py", line 305, in <module>
    videowriter.addBag(bagfile)
  File "/media/ziwei/PortableSSD/rosbag/mp/mp_working/rosbag2video.py", line 273, in addBag
    cv2.imshow(topic, cv_image)
cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'imshow'
> Overload resolution failed:
>  - mat is not a numpy array, neither a scalar
>  - Expected Ptr<cv::cuda::GpuMat> for argument 'mat'
>  - Expected Ptr<cv::UMat> for argument 'mat'"

 S2 

 ``````

 #### idea2. use the files such as bag2png_depth.py imported from the web browser

 ``````
 /--weblink: --*O1. https://idorobotics.com/2021/03/08/extracting-ros-bag-files-to-python/
          #-- to get the code for bag2csv and bag2png
          Steps: S1
          --O2. https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9
          #-- to get the file bag_to_images.py and their Q&A
          --O3. https://coderwall.com/p/qewf6g/how-to-extract-images-from-a-rosbag-file-and-convert-them-to-video
          #-- get the rosrun command to extract the frames into one folder
          --O4. http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data
          #-- to use the export_bad.launch to get the frames extracted
 /--bug1: no rostopic "/file_version"
          --O5.  

 file_link: --1. /media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_recording/25_bag2png.py
            --2. /media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_recording/bag2png_depth.py
 
 ``````

### 3. -- to get the lane detected

``````
/--weblink: --01. https://www.youtube.com/watch?v=mXH1u885bn8
             
         #-- youtube video for duckietown lane following demo
         --O2. https://github.com/duckietown/sim-duckiebot-lanefollowing-demo/blob/master/custom_line_detector/include/line_detector/line_detector2.py
         #-- the python file line_detector2.py which is for line detection of duckietown lane following demo
         --O3 https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
         #-- with code to detect thin white lines
``````

#### idea1. use Hough Transformation
/--weblink: --O1. 
 
 ##### idea1.1 use part by part to draw the lines

 ```
  /--weblink: --O1.
  /--bug1: python3 gb_HSV.py 
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_lane_detection/gb_HSV.py", line 85, in <module>
    for line in lines:
TypeError: 'NoneType' object is not iterable
   /--reason1: too small roi causes no line to draw?
   /--sol1: 
   /--bug2: no right lines cover up the whole lane for the lanes on the four corner?
   /--bug3: cover too much than the original lane, e.g. for the xie lines

```
#### idea2. use HSV detection?
    /-- 
#### idea3. use YOLOv5



/--weblink: 
```
/*--O1 https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
             https://www.youtube.com/watch?v=MdF6x6ZmLAY
         #-- to use ROBOflow to do annotations and get the dataset in YOLOv5 format
         --O2. https://www.youtube.com/watch?v=fu2tfOV9vbY
         #-- How to detect in a local mp4 file with YOLOv5
         tmd: 
         --e.g.1. python3 detect.py --weights yolov5l.pt --source /media/ziwei/yuankai/RGB_video/mp/mp_cereal/mp_MB_3_2022-10-19-19-45-30_camera_color_image_raw_compressed.mp4 --view-img
         #-- the terminal command for detection and also store the data in runs/detect/exp11
         --e.g.2. python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source duckiebot/datasets/images/train
         #-- the tmd to detect duckiebots on a custom dataset created by myself
         /--file_location: the output are located in
         --O1 /media/ziwei/PortableSSD/Junpeng/from_git/yolov5/runs/detect/exp14
         #-- frames detected by the duckie dataset
         --O2 /media/ziwei/PortableSSD/Junpeng/from_git/yolov5/runs/detect/exp16
         #-- image files of the first bot moving detection with frames
         --O3 /media/ziwei/PortableSSD/Junpeng/from_git/yolov5/runs/detect/exp19/my_video-11.mp4
         #-- the dir for the second bot moving video
           



         */--O3. https://blog.csdn.net/m0_53392188/article/details/119334634?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169376041016800213066318%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169376041016800213066318&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-119334634-null-null.142^v93^control&utm_term=yolov5%E4%BF%9D%E5%A7%86&spm=1018.2226.3001.4187
         #-- tutorial in YOLO about how to train in YOLO and detect
         --O4 https://www.hindawi.com/journals/misy/2022/1828848/
         #-- a demo with yolo_txt figure for data visualisation

         */--O5 https://blog.ovhcloud.com/object-detection-train-yolov5-on-a-custom-dataset/
         #-- the english version of the yolov5 tutorial (for tumwiki)




```


/--bug1:
```
 no labels found by running train.py
```

/--sol1: 
```
--1. check whether all json files have been converted to txt format.
         --2. check whether whether the path inside the json2txt.py has a "/" when trying to append a directory for txt files
           e.g. txt_name = '/media/ziwei/PortableSSD/Junpeng/from_git/yolov5/duckiebot/datasets/labels/train/' + json_name[0:-5] + '.txt'
```           
            

#### idea4. draw Hough lines based on the result of canny detection


/-- bug1: no ROI8
/-- sol1: use the file inside duckietown_cv instead of duckietown_cv_lislab






/--bug1:

```
 pip install pyqt5
Collecting pyqt5
  Downloading PyQt5-5.15.9-cp37-abi3-manylinux_2_17_x86_64.whl (8.4 MB)
     |████████████████████████████████| 8.4 MB 5.4 MB/s 
Collecting PyQt5-Qt5>=5.15.2
  Downloading PyQt5_Qt5-5.15.2-py3-none-manylinux2014_x86_64.whl (59.9 MB)
     |████████████████████████████████| 59.9 MB 100 kB/s 
Collecting PyQt5-sip<13,>=12.11
  Downloading PyQt5_sip-12.12.2-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl (335 kB)
     |████████████████████████████████| 335 kB 38.9 MB/s 
Installing collected packages: PyQt5-sip, PyQt5-Qt5, pyqt5
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
spyder 5.1.5 requires pyqtwebengine<5.13, which is not installed.
spyder 5.1.5 requires pyqt5<5.13, but you have pyqt5 5.15.9 which is incompatible.
```
/--susp1: pip install 'PyQt5<5.13'


/--result1: the result best.pt and last.pt have been stored into the '/media/ziwei/PortableSSD/Junpeng/from_git/yolov5/runs/train/exp4/weights/best.pt'
/--result2: the result of detection of duckiebot has been stored into '/media/ziwei/PortableSSD/Junpeng/from_git/yolov5/runs/detect/exp14
'


#### idea5. detect and draw contours based on canny detection

##### file1. pilot_lane_detection.py

/--bug1:

```
[ WARN:0@0.009] global loadsave.cpp:244 findDecoder imread_('/home/junpeng/Documents/to_git/duckietown_cv/images/for_detection/frame02.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_lane_detection/pilot_lane_detection.py", line 22, in <module>
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.error: OpenCV(4.7.0) /io/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'


```

/--reason1: 

```
wrong or no directory
as the PC can be different and accordingly also the directory

```

/--sol1:
```
change the directory to the local PC
e.g. image= cv2.imread('/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/images/for_detection/frame06.jpg')

```



#### idea6. test other gits 

 ##### 1. AdvancedLaneFinding.py
 #-- to
 /--bug1: inotify_add_watch(...) failed: "No space left on device
 /--sol1: in gbt Q15
  S1. 

 /--bug2:
    Traceback (most recent call last):
    File "/home/ziwei/anaconda3/lib/python3.9/site-packages/moviepy/video/VideoClip.py", line 262, in write_videofile
        codec = extensions_dict[ext]['codec'][0]
    KeyError: 'mkv'

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_lane_detection/AdvancedLaneFinding.py", line 547, in <module>
        output_clip.write_videofile(video_output, audio=False)
    File "<decorator-gen-55>", line 2, in write_videofile
    File "/home/ziwei/anaconda3/lib/python3.9/site-packages/moviepy/decorators.py", line 54, in requires_duration
        return f(clip, *a, **k)
    File "<decorator-gen-54>", line 2, in write_videofile
    File "/home/ziwei/anaconda3/lib/python3.9/site-packages/moviepy/decorators.py", line 135, in use_clip_fps_by_default
        return f(clip, *new_a, **new_kw)
    File "<decorator-gen-53>", line 2, in write_videofile
    File "/home/ziwei/anaconda3/lib/python3.9/site-packages/moviepy/decorators.py", line 22, in convert_masks_to_RGB
        return f(clip, *a, **k)
    File "/home/ziwei/anaconda3/lib/python3.9/site-packages/moviepy/video/VideoClip.py", line 264, in write_videofile
        raise ValueError("MoviePy couldn't find the codec associated "
    ValueError: MoviePy couldn't find the codec associated with the filename. Provide the 'codec' parameter in write_videofile.
sol2: 


### 4. How to detect the middle line between two lanes

#### idea1 use opencv methods
 /--weblink: O1 https://stackoverflow.com/questions/64396183/opencv-find-a-middle-line-of-a-contour-python
          #-- contains some code for detecting the middle line of a contour
          O2 gbt Q26
          #-- use the hoi with height and width in a half and then use Hough transform to draw the lines 
          O3 

  #### idea 2 use the cv2.boundingRect(cnt)


### 5. How to get the speed of the bot

#### idea1 use optical flow obtained by depth information

 /--weblink: 
 `````` 
 --O1 gbt CV project plus Q61 
 --O2
  
 1.
 https://blog.csdn.net/weixin_45303602/article/details/133814463?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%85%89%E6%B5%81%E6%B5%8B%E9%80%9F&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-133814463.142^v96^control&spm=1018.2226.3001.4187
 ## with functions in opencv to implement optical flow

 *2.
 https://blog.csdn.net/uncle_ll/article/details/121835741?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%85%89%E6%B5%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-121835741.142^v96^control&spm=1018.2226.3001.4187
 #-- code in opencv about the sparse and dense optical flow implementation
 
  


 `````` 
 /--bug1: pyrealsense2 package missing by vscode
 /--susp1: conda install -c conda-forge librealsense
   //-- bug1: can not be installed
 /--sol1: pip install pyrealsense2
 /--

#### idea2 simply use YOLOv5 to detect and estimate

 /--weblink:
 ```
 *--O1. literature_dir: '/home/junpeng/Documents/to_git/duckietown_cv_lislab/literature/pose&speed_estimation/vision_based...pdf'
          #-- an essay with speed estimation
  //--chapter: IV. VEHICLE SPEED ESTIMATION BASED ON RNN WITH YOLO BOUNDING BOXES
   k1. deep learning method
   k2. fine-tuning of YOLO v5m on the Vehicles-OpenImages dataset
   **--k3. extract three features to be used for vehicle speed estimation, equivalent to coordinates of the lower-left and upper-right vertices of the bounding box 
    --k3-1. bounding box area
    --k3-2. x-coordinate of the lower left vertex and
    --k3-3. y-coordinate of the lower left vertex
   --k4 advantage the proposed methods
       e.g.  it does not require prior knowledge of real-world dimensions such as vehicle size, road width, camera distance and angle in relation to the road.
    ///--fig
    O1 Fig. 3. Features extracted from video useing YOLO
    #-- with the frame plot of three features mentioned above

 --O2. https://github.com/ultralytics/yolov5/issues/10510
  #-- a git forum for the Q&A of the speed estimation and in/out of an area
   /-- susp1: Simple Inference Example from glenn-jocher

  --O3. https://www.youtube.com/watch?v=E1ZNbsIr6Bo
  #-- youtube video for the visualisation of the speed detection and also the ID 
  
  --O4. https://ieeexplore.ieee.org/document/9486316
  #-- distance measurement with an essay in IEEE

  *--O5 https://medium.com/axinc-ai/yolov5-the-latest-model-for-object-detection-b13320ec516b
  #-- with information about how to get the output of YOLO in txt format to get data such as width of bbox and locations of bots

  *--O6 

 ```

#### idea3. use the ones from intelligent traffic systems

/--weblink:
```
 --O1.
 https://stackoverflow.com/questions/43732185/importerror-no-module-named-easydict
 #-- to install the missing modules such as to initialize the module deep_sort 

```

 ##### issue1. how to get the txt format with coordinate data of bots

 *O1. use the detect.py and change the save_txt option while running that in terminal


#### idea4. use the Apriltag for speed estimation

/--weblink:
```
--O1.



```

#### use optical flow 
/--weblink:
```
O1. 
https://github.com/swhan0329/vehicle_speed_estimation
#-- to get the speed of vehicles on high way for speed estimation in a certain moment


```

 ### 6. How to realize the real-time detection

#### idea1 using YOLOv5

 ```
  O1.
  python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source 0
  #-- --source 0 means that it is for the webcam

  /--bug1. no module named torch, trackback when importing torch on the main PC of LIS
  /--susp1: 
  
  
 
 ```


 ### 7. How to get the segmentation (e.g. instance segmentation) of the lanes

 #### idea1. use instance segmentation

 ```
 /--weblinks: 
 O1. https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-020-00172-z
 #-- essay about instance segementation using CNN methods
 with literature_dir: '/home/junpeng/Documents/to_git/duckietown_cv_lislab/literature/instance_segmentation/CNN based....pdf'

 O2. https://github.com/facebookresearch/detectron2
 #-- the github page of the detectron2
 with literature_dir: ''

 
 ```

 #### idea2. use lane assist

 ```
 /--weblinks: 
  --O1. https://github.com/topics/lane-keeping-assistant
 #-- contains three gits about the lane assist
   --O1-1 https://github.com/A-Raafat/Torcs---Reinforcement-Learning-using-Q-Learning
   #-- git with figures showing the scene from upside down
   --O1-2 
 *--O2. https://junshengfu.github.io/driving-lane-departure-warning/
 #-- git with great visualisation of the segmented areas, but in the auto's sites instead of the warped site that we want
   --O2-1 https://www.youtube.com/watch?v=fqQFVK4ZxoQ
   --O2-2 https://www.youtube.com/watch?v=3-CMwxaScEo
   #--demo videos with the segmented regions change colors

 *--O3. https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-
  #-- git with visualisation of the segmented areas, but in the auto's sites instead of the warped site that we want



 ```

 #### idea3. use semantic segmentation

 /--weblinks:

 ```
 --O1. https://github.com/kulkarnikeerti/SegNet-Semantic-Segmentation
  ## git for SegNet
 --O2. https://www.mrt.kit.edu/z/publ/download/2018/Meyer2018SemanticLanes.pdf
  #-- essay for lane segmentation
 --O3. 
 
 
 ```


## 2. MDLHOI dataset project

### 1. to generate the mdlhoi dataset

#### js_generator_double.py

/--bug1:
Intel MKL FATAL ERROR: Cannot load /home/ziwei/anaconda3/lib/python3.9/site-packages/mkl/../../../libmkl_rt.so.1.

/--sol1: 
S1. reinstall mkl with 1. "conda uninstall --force mkl mkl-service"
                       2. "conda install mkl mkl-service"
S2. restart the IDE




/--bug2:

Traceback (most recent call last):
  File "/media/ziwei/yuankai/code/js_generator_double_mp.py", line 247, in <module>
    GTlist.sort(key=lambda x:int(x[6:-4]))
  File "/media/ziwei/yuankai/code/js_generator_double_mp.py", line 247, in <lambda>
    GTlist.sort(key=lambda x:int(x[6:-4]))
ValueError: invalid literal for int() with base 10: ''

sol1: 1. Do not remove something in cvat e.g. mp_pick_up_4 works well



### 2. to generate a video with HOI segment

#### file1: demo_double_anno_time.py

/--bug1.

```
Traceback (most recent call last):
  File "/media/ziwei/yuankai/code/demo_double_anno_time.py", line 129, in <module>
    frame_action_next_2.append(To_print[key_number_2][2])
IndexError: list index out of range

```

/--sol1: 

```
 current_actuation_time_2.append(((slice_change_2-i)/slice_change_2)*slices_time[key_number_2-1])
        next_actuation_time_2.append(slices_time[key_number_2])
        third_actuation_time_2.append(slices_time[key_number_2+1])
        frame_action_now_2.append(To_print[key_number_2-1][2])
        frame_object_now_2.append(To_print[key_number_2-1][3])
        if len(To_print[key_number_2])>2 and len(To_print[key_number_2+1])>2:
            frame_action_next_2.append(To_print[key_number_2][2])
            frame_object_next_2.append(To_print[key_number_2][3])
            frame_action_third_2.append(To_print[(key_number_2)+1][2])
            frame_object_third_2.append(To_print[(key_number_2)+1][3])
        else:
            frame_action_next_2.append(To_print[key_number_2][0])
            frame_object_next_2.append(To_print[key_number_2][1])
            frame_action_third_2.append(To_print[(key_number_2)+1][0])
            frame_object_third_2.append(To_print[(key_number_2)+1][1])


```


/--bug2: overlap of put Texts

/--sol1: 
```
to use the value after the '<' in the () of cv2.putText

e.g.
cv2.putText(image,"current HOI segment: "+"<" +"Human 2, " + action_label[frame_action_now_2[j]]+" ,"+object_label[frame_object_now_2[j]]+" ,"+str(format(current_actuation_time_2[j]))+" >",(0,240), font,0.75, (0,0,255),2)

```

### 3. to extract depth png files from the bag files

#### file1: bag2png_depth.py

/--bug1: 

```
Traceback (most recent call last):
  File "/media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_recording/bag2png_depth.py", line 20, in <module>
    next_timestamp = bag.get_start_time()
  File "/opt/ros/noetic/lib/python3/dist-packages/rosbag/bag.py", line 802, in get_start_time
    raise ROSBagException('Bag contains no message')
rosbag.bag.ROSBagException: Bag contains no message


```

/--sol1: check the data integrity in the bag file (rosbag info <>)


# Troubleshooting

## E1. no playable streams after composing a video


```
/weblink:
https://stackoverflow.com/questions/29320976/opencv-videowrite-doesnt-write-video

/sol1: instead of defining the width&height of the output video myself, just use the ones from the captured video to make sure that they are all compatible for display.



```

## E2. convert cvtColor error

```
/weblink:

/sol1:
wrong or missing directory check directory integrity



```


