# Keyboard

1.
Ctrl+Shift+N
 --- to open a new folder in a folder



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
size:        865.0 KB **#-- to get the size information**
messages:    12471 **#-- to get the rosmsg infomation**
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
 #-- to play back in a loop without stopping
 O1 rosbag play -l
 
3.
rosbag record 
 O1 rosbag record -a

``````


## 5.2 rostopic

```
1.
rostopic list
 O1 rostopic list -v
 #-- to get the publischers and subscribers
 O2
 

 
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

## 7.1 Duckietown

```



```

