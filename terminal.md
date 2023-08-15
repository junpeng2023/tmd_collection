# Keyboard

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


## in vscode

1.
Ctrl+P 
 --- to shift quickly between different files in the same folder in vscode(otherwise too slow)

2.
Ctrl+O
 --- to open a new file

3.
Ctrl+B
 --- to get the left side dictionaries appear or disappear





# Find



## 1.1 How to find something
```

O1 which <package>

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
cat id_rsa.pub
#-- to get the ssh key of rsa to paste that into the git(within the settings and "new ssh-key")



```

# 4. Anaconda

```
conda init

```

# 5.ROS
```

1.
roscd <>
 #-- to get to a ros package quickly

2.
rosls <>
 #-- to list what is inside the ros package quickly

3. 
printenv | grep ROS
 #-- to get version information
 
roslaunch panda_moveit_config demo.launch

rosrun <> <>
 ## turtle (from rosbag/Tutorials/Recording and playing back data )
  O1 rosrun turtlesim turtlesim_node
   #-- to get the visualization of turtle
  O2 rosrun turtlesim turtle_teleop_key
   #-- to use keyboard to control the turtle
  O3 
  

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
 

 
```




# 6. Python

## YOLO

```

1.
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt # aim: to train with & 1.--img 2.--epochs 3.--data 4.--weights 
#-- detect.py runs YOLOv5 inference on a variety of sources, downloading models automatically from the latest YOLOv5 release, and saving results to runs/detect. Example inference sources are:

python detect.py --source 0  # webcam
                          img.jpg  # image 
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                         'path/*.jpg'  # glob
                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

2.
python3 <>

3.
!python val.py --weights yolov5s.pt --data coco.yaml --img 640 --half
#-- to validate a model's accuracy on the COCO dataset's val or test splits.


```


# 7. Docker

``
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
docker run ubuntu

docker run -it ubuntu


5.





``

# 8. Duckietown


```
1.
dts duckiebot demo --duckiebot_name ![DUCKIEBOT_NAME] --demo_name ![DEMO_NAME] --package_name ![PACKAGE_NAME] --image gitlab.lrz.de:5005/tum-lis/staff/projects/duckietown/base:latest
 #-- to start the DEMO_NAME.launch launch file in the PACKAGE_NAME package from the gitlab.lrz.de:5005/tum-lis/staff/projects/duckietown/base:latest Docker image on the DUCKIEBOT_NAME Duckiebot. 

2. dts fleet discover
 #-- to show available Duckiebots

3. dts duckiebot shutdown DUCKIEBOT_NAME
 #-- to shut down duckiebots(otherwise always open)

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

 ``````

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





