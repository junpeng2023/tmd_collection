

# OpenCV

# 1. ffmpeg

## 1.1 How to convert png files into a mp4
  ``````

  ffmpeg -r <frame_rate> -i <png_sequenz> -c:v libx264 -pix_fmt yuv420p <path_to_mp4>

  O1
   ffmpeg -r 30 -i %04d.PNG -c:v libx264 -pix_fmt yuv420p /media/ziwei/yuankai/pictures_to_show/videos/9_pick_up/col/col_pick_up_1/anno_col_pick_up_1.mp4

  ``````
 ### Troubleshooting
  .PNG instead of .png otherwise unable to use ffmpeg to do the convertion with the output of "Could find no file with path '%04d.png' and index in the range 0-4
%04d.png: No such file or directory"

## 1.2 How to convert mp4 files into png files
 ``````

 ffmpeg -i <mp4_file> -r <frame_rate> frame%d.png

 O1 
 ffmpeg -i col_bag_1_depth-2023-07-13_22.08.57.mp4 -r 30 /media/ziwei/PortableSSD/depth_video/images/col/col_bad/col_bad_1/frame%4d.

 ``````

 ### Troubleshooting
   1. please check the frame rate and ensure it is right when telling the ffmpeg
   2. no "-o" for output video


# 2. camera_calibration

 ### Troubleshooting
  bug: How many grids for detection? 
  S: (x-1,y-1) for x,y are the length and width of the chessboard, otherwise the grids may not be detected






# n. link collection


## n.1 github
``````






``````