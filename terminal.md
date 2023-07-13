# Keyboard

1.
Ctrl+Shift+N
 --- to open a new folder in a folder

2.
F11
 --- to get or exit full screen modus otherwise unpossible to exit

3, 


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

```

# 2. Installation of something

# 3.GIT

```
git branch
 output: *main

git add
 
 O1 git add --all
  #-- to add all files modified 

git commit -m ""

git push

 O1 git push origin main 

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

2.
python3 <>

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
