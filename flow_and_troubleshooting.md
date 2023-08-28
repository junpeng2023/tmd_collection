# Flow of projects

## CV_project

### 1. -- to get the pcl Pointcloud with given depth and rgb frames

#### idea1. -- How to install the pcl package?

weblink:
1. https://tony-laoshi.github.io/pcl%E5%BA%93%E5%9C%A8ubuntu20.04%E4%B8%8A%E7%9A%84%E5%AE%89%E8%A3%85/
#-- with how to install all dependencies

S1. sudo apt install libpcl-dev

S2. installing the python pcl
 O1 pip install python-pcl -vvvv
 bug1: errors with pcl common
  sol1: pip install python-pcl -vvvv
  sol2: 

S3. to install the python3-pcl instead of python-pcl
 weblink: https://blog.csdn.net/zsssrs/article/details/120054425
 #-- how to get the package into anaconda
 S3-1 use "sudo apt-get install python3-pcl"

S4




#### idea2. -- to get the docker image for pcl and run on it

bug1: docker image can not be run


### 2. -- to get the rgb and depth frames from one given rosbag file

#### idea1. use the file rosbag2video.py 
 weblink: https://github.com/mlaiacker/rosbag2video/tree/master
 #-- git repository for rosbag2video.py
 S1 use the terminal command such as "python3 rosbag2video.py --fps 25 --rate 1 -t /camera/aligned_depth_to_color/image_raw mpmp_working_2_2022-10-19-20-26-59.bag "
 bug1: can not use -s in the tmd, otherwise with error message
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

 #### idea2. use the files such as bag2png_depth.py imported from the web browser
 weblink: https://idorobotics.com/2021/03/08/extracting-ros-bag-files-to-python/
          #-- to get the code for bag2csv and bag2png

 file_link: 1. /media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_recording/25_bag2png.py
            2. /media/ziwei/PortableSSD/Junpeng/to_git/duckietown_git/duckietown_cv/code_recording/bag2png_depth.py
 
 
#### 3. -- to get the lane detected

#### idea1. use Hough Transformation

#### idea2. use HSV detection?

#### idea3. 