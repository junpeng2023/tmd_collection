# Keyboard

## to operate the modus of the pc

```
1.
window_key+l
 #-- to swich off the PC to the sleeping modus
 

```

## in Files

```
1.
Ctrl+Shift+N
 #-- to open a new folder in a folder

2.
F11
 #-- to get or exit full screen modus otherwise unpossible to exit

3, 
Ctrl+A
 #-- to get everything inside the file selected

4.
Entf/delete
#-- to move files to trash quickly

```
 

## in vscode

```
1.
Ctrl+P 
 --- to shift quickly between different files in the same folder in vscode(otherwise too slow)

2.
Ctrl+O
 --- to open a new file

3.
Ctrl+B
 --- to get the left side dictionaries appear or disappear

4.
Ctrl+Shift+V
 #-- to open the preview of a .md file

5. 
Ctrl+'+'
 #-- to zoom in the whole window of vscode

6.
Ctrl+'-'
 #-- to zoom out the whole window of vscode

```

## in web browser

```
1.
Ctrl+P
 #-- to print something and show the print window as the first step

2.
right-klick + open in a new tab
 #-- to open a new tab without covering the current tab

```

# Find



## 1. How to find something
```

O1 which <package>
e.g. 1. which dts
#-- to find the package dts installed or not and find the path where it is installed
e.g. 2. which rviz
#-- to find the package rviz installed or not and find the path where it is installed


O2 rosls <>

O3 rospack find <>

4.
sudo find / -name <#file_name>
#-- to get the absolute path of a file in all places, otherwise we do not know where the file is

O1 sudo find / -name demo_30fps-1.mkv

5.
pwd

#-- to get the path to a file



```

## 1.1 How to find the path of a file

```
1.
S1 #-- use readlink to get the path of a file better and faster than pwd
cd <target_directory>

S2 
readlink -f <file_name>

2. #-- to get the absolute directory of a file
realpath <file_name>





```

# 2. Installation of something

# 3.GIT

```
git branch
 output: *main

git checkout <>
#-- to move to another branch 
 A1 git checkout -b
 #-- to create a branch and move to that

git add
 
 O1 git add --all
  #-- to add all files modified 

git commit -m ""

git push

 O1 git push origin main 

git reset <> or git reset
#-- to undo "git add <>"

```

## 3.1 ssh-key for git clone
```

/// git 

1.
ssh-keygen
 #-- to generate a ssh purblic key

// output: 
Generating public/private rsa key pair.
Enter file in which to save the key (/Users/rahulwagh/.ssh/id_rsa):
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/rahulwagh/.ssh/id_rsa.
Your public key has been saved in /Users/rahulwagh/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:Okq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWA rahulwagh@local
The key's randomart image is:
+---[RSA 3072]----+
|.ooo..+oo.       |
| oo o..o+.       |
|=E = = +.        |
|*oo X o .        |
|.+ = o  S        |
|o.  + ..         |
|o ..+=+          |
| o + *++         |
|. . o.+.         |
+----[SHA256]-----+


2. 
/// cd to  e.g. /Users/rahulwagh/.ssh/ with id_rsa.pub as the first step
cd ~/.ssh

3.
O1
cat id_rsa.pub
#-- to get the ssh key of rsa to paste that into the git(within the settings and "new ssh-key")

O2
cat ~/.ssh/id_rsa.pub
#-- to get the ssh key of rsa to paste that into the git(within the settings and "new ssh-key") without cd to .ssh first

4.
# copy&paste into git in settings-> SSH and GPG keys




```

# 4. Anaconda

```

1.
conda init

2.
conda activate base
#-- to activate a environment

3.


```

# 5.ROS
```

1.
roscd <>
 #-- to get to a ros package quickly
 e.g. 1. 
 roscd usb_cam
 e.g. 2.
 roscd camera_calibration
 e.g. 3.
 

2.
rosls <>
 #-- to list what is inside the ros package quickly

3. 
printenv | grep ROS
 #-- to get version information
 
4.
roslaunch <package> <launch_file>
e.g 1.
roslaunch panda_moveit_config demo.launch
e.g. 2.
roslaunch usb_cam usb_cam-test.launch
 bug1:  {
          ROS_MASTER_URI=http://localhost:11311

        process[usb_cam-1]: started with pid [104971]
        process[image_view-2]: started with pid [104972]
        [ INFO] [1691682165.921051114]: Initializing nodelet with 16 worker threads.
        [ INFO] [1691682165.979152625]: Using transport "raw"
        [ INFO] [1691682165.989513159]: using default calibration URL
        [ INFO] [1691682165.989916393]: camera calibration URL: file:///home/ge78pav/.ros/camera_info/usb_cam.yaml
        [ INFO] [1691682165.991087210]: Starting 'usb_cam' (/dev/video0) at 640x480 via mmap (yuyv) at 30 FPS
        [ WARN] [1691682165.991120775]: /dev/video0 does not support setting format options.
        [ WARN] [1691682165.991131501]: /dev/video0 supports: 
          Width/Height 	 : 1280/720
          Pixel Format 	 : MJPG
        [ERROR] [1691682165.991145483]: VIDIOC_S_FMT error 16, Device or resource busy
        [usb_cam-1] process has died [pid 104971, exit code 1, cmd /opt/ros/noetic/lib/usb_cam/usb_cam_node __name:=usb_cam __log:=/home/ge78pav/.ros/log/b12f8a7a-377a-11ee-a6e5-ddb6416cc2d7/usb_cam-1.log].
        log file: /home/ge78pav/.ros/log/b12f8a7a-377a-11ee-a6e5-ddb6416cc2d7/usb_cam-1*.log
 }
  solution 1:
   # ?
 

5.
rosrun <> <>
 ## turtle (from rosbag/Tutorials/Recording and playing back data )
  O1 rosrun turtlesim turtlesim_node
   #-- to get the visualization of turtle
  O2 rosrun turtlesim turtle_teleop_key
   #-- to use keyboard to control the turtle
  O3 rosrun image_view extract_images _sec_per_frame:=0.01 image:=<IMAGETOPICINBAGFILE>
   #-- to extract frames from one rosbag file
  O4 

6. 
#-- to tell the ROS that changes have been made in the workspace 

 S1. 
 cd <workspace_directory>
 e.g 1. 
 cd catkin_ws

 S2.
 catkin_make
 #-- to do the make command designed for ROS so that many steps can be merged into one step.

 S3.
 e.g. 1.
 source devel/setup.bash  
 e.g. 2.
 source /opt/ros/noetic/setup.bash



```


## 5.1 rosbag

``````
1.
rosbag info <>
//output: e.g.
path:        2014-12-10-20-08-34.bag 
version:     2.0
duration:    1:38s (98s)
start:       Dec 10 2014 20:08:35.83 (1418270915.83)
end:         Dec 10 2014 20:10:14.38 (1418271014.38)
size:        865.0 KB                                                     **#-- to get the size information**
messages:    12471                                                        **#-- to get the rosmsg infomation**
compression: none [1/1 chunks]
types:       geometry_msgs/Twist [9f195f881246fdfa2798d1d3eebca84a]
             rosgraph_msgs/Log   [acffd30cd6b6de30f120938c17c593fb]
             turtlesim/Color     [353891e354491c51aabe32df673fb446]
             turtlesim/Pose      [863b248d5016ca62ea2e895ae5265cf9]
topics:      /rosout                    4 msgs    : rosgraph_msgs/Log   (2 connections)
             /turtle1/cmd_vel         169 msgs    : geometry_msgs/Twist
             /turtle1/color_sensor   6149 msgs    : turtlesim/Color
             /turtle1/pose           6149 msgs    : turtlesim/Pose
//

2.
rosbag play <>
 
 O1 rosbag play -l
  #-- to play back in a loop without stopping
 O2 rosbag play -r <play_speed> <bag_name>
  #-- to play back at a certain speed

3.
rosbag record 
 O1 rosbag record -a
 O2 rosbag record -O subset <topic1> <topic2>
  #-- to tell the bag to only subscribe certain topics
 O3 rosbag record --duration=30 --output-name=/tmp/mybagfile.bag \
    /topic1 /topic2 /topic3


``````


## 5.2 rostopic

```
1.
rostopic list
 O1 rostopic list -v
 #-- to get the publischers and subscribers
 O2 rostopic echo <>
 
2.
rostopic echo <topic_name>

3.
rostopic info <topic_name>


 
```




# 6. Python

## YOLO

```

1.
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt # aim: to train with & 1.--img 2.--epochs 3.--data 4.--weights 
#-- detect.py runs YOLOv5 inference on a variety of sources, downloading models automatically from the latest YOLOv5 release, and saving results to runs/detect. Example inference sources are:
 P1 --weights
  O1 yolov5n.pt
  O2 yolov5m.pt
  O3 yolov5l.pt

O1 python3 train.py
#-- to run the script after getting all the files(e.g. .yaml files) changed


2,
python detect.py --source 0  # webcam
                          img.jpg  # image 
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                         'path/*.jpg'  # glob
                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

O1 python3 detect.py --weights yolov5l.pt --source /media/ziwei/yuankai/RGB_video/mp/mp_cereal/mp_MB_3_2022-10-19-19-45-30_camera_color_image_raw_compressed.mp4 --view-img
#-- to detect a video in the local directory with the default dataset with window to show the result of detection while running the code

O2 python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source duckiebot/datasets/images/train
#-- to detect some frames inside the local directory with the custom dataset created by myself using labelme

O3 python3 detect.py --weights runs/train/exp7/weights/best.pt --data duckiebot/duckiebot_parameter.yaml --source 0
#-- to detect realtime with the camera using the custom dataset created

O4 1.
 python3 detect.py --weights yolov5l.pt --source /media/ziwei/yuankai/RGB_video/mp/mp_cereal/mp_MB_3_2022-10-19-19-45-30_camera_color_image_raw_compressed.mp4 --view-img
 #-- the terminal command for detection of a local video, show the video in a window and also store the data in runs/detect/exp11
 
/--bug1: Do not choose the --view-img when trying to detect on the image files, as we will get endless windows shown at the same time leading to a disastrous stuck



3.
python val.py --weights yolov5s.pt --data coco.yaml --img 640 --half
#-- to validate a model's accuracy on the COCO dataset's val or test splits.
 P1 


```


# 7. Docker

```
1.
docker ps
 #-- to list all running containers

docker container list 
 #-- alternative of "docker ps"

docker container list -a 
 #-- to see all containers including running and stopped ones




2.
docker-compose up -d 
 #-- to update the docker after any changes

3.
docker logs <#container_name>
 ---
 O1 docker logs cvat_server
 O2 


4.
docker run <>
docker run ubuntu

docker run -it ubuntu

docker run hello-world
docker run --rm hello-world




5. 
docker image list
#-- to show all docker images activate


6.
docker container list
#-- to show all docker container activate



```

## 7.1 to install docker for duckietown

```
link: https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_laptop/setup_docker.html


1.
sudo apt-get remove docker docker-engine docker.io containerd runc
 #-- to ensure that the older versions of docker would not interfer and 2. get the docker.io

error1: docker-engine can not be located
solution:?
link:
https://stackoverflow.com/questions/39645118/docker-unable-to-locate-package-docker-engine

6.
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg
 #-- to install 

7.
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
#-- to add official keys

8.
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
#-- to set up the docker repository

9.
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get install docker-compose
#-- install Docker Engine and Docker Compose

10.
sudo adduser `whoami` docker
#-- Start by adding the user “docker” to your user group, then log out and back in

11.
docker run hello-world
 #-- to run the docker with the demo "hello world"

error1: 
docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
See 'docker run --help'.
solution1: sudo chmod 666 /var/run/docker.sock
link: https://blog.csdn.net/qq_35885175/article/details/122316520

12.
docker --version
docker buildx version
#-- Make sure the Docker version is v1.4.0+ and buildx version v.0.8.0+




```

# 8. Duckietown


```
1.
dts duckiebot demo --duckiebot_name ![DUCKIEBOT_NAME] --demo_name ![DEMO_NAME] --package_name ![PACKAGE_NAME] --image gitlab.lrz.de:5005/tum-lis/staff/projects/duckietown/base:latest
 #-- to start the DEMO_NAME.launch launch file in the PACKAGE_NAME package from the gitlab.lrz.de:5005/tum-lis/staff/projects/duckietown/base:latest Docker image on the DUCKIEBOT_NAME Duckiebot. 

2. dts fleet discover
 #-- to show available Duckiebots

3. dts duckiebot shutdown DUCKIEBOT_NAME
 #-- to shut down duckiebots(otherwise always open)

4. dts

5. #-- to avoid errors when starting the dts
 S1. cd ~/.dt-shell
 S2. rm -rf <commands_folder>
  e.g.1. rm -rf commands-multi
 



```

## 8.1 duckietown-visualization


### 8.1.1 Installation
``````

1.
/// cd to /src of our workspace(e.g. /catkin_ws/src) as the first step
git clone https://github.com/duckietown/duckietown-visualization


2.
catkin build 
/// Run catkin_make instead if you don't use python-catkin-tools.

3. 
source devel/setup.zsh
#-- 在创建了ROS的workspace后，需要将workspace中的setup.bash文件写入~/.bashrc 文件中，让其启动

``````

### 8.1.2 Running the map visualization

``````

1.
roslaunch duckietown_visualization publish_map.launch
#-- to run the visualization of the default map robotarium1

2. 
roslaunch duckietown_visualization publish_map.launch map_name:="small_loop"

3.
roslaunch duckietown_visualization publish_map.launch map_name:="small_loop" rviz_config:="path/to/myconfig.rviz"



``````



# 9 CV

## 9.1 ffmpeg

``````
  1.
  ffmpeg -r <frame_rate> -i <png_sequenz> -c:v libx264 -pix_fmt yuv420p <path_to_mp4>
  #-- to convert some png files into a mp4 video
  O1
   ffmpeg -r 30 -i %04d.PNG -c:v libx264 -pix_fmt yuv420p /media/ziwei/yuankai/pictures_to_show/videos/9_pick_up/col/col_pick_up_1/anno_col_pick_up_1.mp4

  A1 
  cat $(ls -v depth_*_normalized.png) | ffmpeg -f image2pipe -r 25 -i - -c:v libx264 -pix_fmt yuv420p output_video.mp4
  #-- to get every single frames to be involved in the mp4 file

  ``````

  ``````
 2.
 ffmpeg -i <mp4_file> -r <frame_rate> frame%d.png
 #-- to extract frames from a mp4 video
 O1 
 ffmpeg -i col_bag_1_depth-2023-07-13_22.08.57.mp4 -r 30 /media/ziwei/PortableSSD/depth_video/images/col/col_bad/col_bad_1/frame%4d.


 3.
 ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4
 #-- to convert the video to a resolution that we want to have such as 1280*720

 4.
 ffmpeg -i <input_image_file> -vf "scale=1280:720" <output_image_file>
 #-- to convert the image file into a resolution that we want to have such as 1280*720

 5.
 ffmpeg -i <input_mp4_file> -filter:v "setpts=<multiplier>*PTS" <output_mp4_file>
 e.g.1.
 ffmpeg -i input.mkv -filter:v "setpts=0.5*PTS" output.mkv
 #-- to adjust the speed of a video


 6.
 ffmpeg -i <original_format> -c copy <target_format>
 #-- to convert a video into another format
 e.g.1.
 ffmpeg -i example.mkv -c copy example.mp4

 ``````

 
 ```
 
 
 ```

 # 10. File processing


 ## 10.1 zip

 ``````

 1.
 zip -r <zip_file> <folder>

 2.
 zip <zip_file> <file>


 ``````

  ## 10.2 File_editor

   
  ### 10.2.1 gedit

  ```
  
  gedit <file_name>
  #-- to create or edit a file
  bug1: an empty file created by gedit will not be stored
   -- solution: to write down something into the file and then close the file for storage

  bug2: forget the "gedit", then waste some time costs

  ```

  ### 10.2.2 nano

  ### 10.2.3 vim


  ### 10.2.4 code .

  ### 10.2.5 cat


# 11. Virtual Environment

 ## 11.1 pipenv

```
 1.
 pipenv install
 #-- to install the virtual environment

 2.
 pipenv shell
 #-- To activate this project's virtualenv, otherwise no virtual environment activated

```

# 12. Jupyter-Notebook

```
  1.
  jupyter-notebook
  #-- to activate the jupyter notebook in a web browser

  2.
  


```

# 13. pip

```
1.
pip install <package>
 A1 pip install `<package><<one_version>`
 e.g.1. pip install 'PyQt5<5.13'





```


