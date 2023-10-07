from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff

import sys
import time
import math

sys.path.insert(0, './yolov5')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def find_accidents(rects):
    is_accident_happen = []
    t = 3000  # 设定阈值，这个值越大，碰撞越不容易检出。设置一个较小的值时，可当做碰撞预警功能来使用。
    crash_index = []
    # 嵌套循环读出任意两个车辆框，进行两两对比，(xmin,ymin)为框的左上角坐标，(xmax,ymax)为框的右下角坐标
    for i in range(len(rects)):
        A_xmin, A_ymin, A_xmax, A_ymax, A_conf, A_class = rects[i]

        for j in range(len(rects)):
            B_xmin, B_ymin, B_xmax, B_ymax, B_conf, B_class = rects[j]

            if A_xmin == B_xmin and A_xmax == B_xmax and A_ymin == B_ymin and A_ymax == B_ymax:
                continue

            # A、B两框没有相交，则跳过，进入下一次循环
            if A_ymax < B_ymin:
                continue

            elif A_xmax < B_xmin:
                continue

            elif B_xmax < A_xmin:
                continue

            elif B_ymax < A_ymin:
                continue

            # 通过A、B框位置关系判断其是否发生碰撞，若相交、存在包含关系，则判定为碰撞，将其索引加入crash_index列表中。
            if (B_xmin < A_xmin + t and A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and (
                    B_ymin < A_ymin + t and A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):  # 01
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (B_xmin < A_xmin + t and A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and (
                    A_ymin < B_ymin and B_ymin < B_ymax and A_ymax < B_ymax):  # 02
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (B_xmin < A_xmin + t and A_xmin + t < B_xmax - t and A_xmax > B_xmax - t) and (
                    A_ymin < B_ymin + t and B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):  # 03
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin and B_xmin < B_xmax and A_xmax > B_xmax) and (
                    B_ymin < A_ymin + t and A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):  # 04
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin + t and B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and (
                    B_ymin < A_ymin + t and A_ymin + t < B_ymax - t and A_ymax > B_ymax - t):  # 05
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin + t and B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and (
                    A_ymin < B_ymin and B_ymin < B_ymax and A_ymax > B_ymax):  # 06
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin + t and B_xmin + t < A_xmax - t and B_xmax > A_xmax - t) and (
                    A_ymin < B_ymin + t and B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):  # 07
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin and B_xmin < B_xmax and A_xmax > B_xmax) and (
                    A_ymin < B_ymin + t and B_ymin + t < A_ymax - t and B_ymax > A_ymax - t):  # 08
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (A_xmin < B_xmin and B_xmin < A_xmax and B_xmax > A_xmax) and (
                    B_ymin < A_ymin and A_ymin < A_ymax and B_ymax > A_ymax):  # 09
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (B_xmin < A_xmin and A_xmin < B_xmax and A_xmax > B_xmax) and (
                    B_ymin < A_ymin and A_ymin < A_ymax and B_ymax > A_ymax):  # 10
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (B_xmin < A_xmin and A_xmin < A_xmax and B_xmax > A_xmax) and (
                    A_ymin < B_ymin and B_ymin < A_ymax and B_ymax > A_ymax):  # 11
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

            elif (B_xmin < A_xmin and A_xmin < A_xmax and B_xmax > A_xmax) and (
                    B_ymin < A_ymin and A_ymin < B_ymax and A_ymax > B_ymax):  # 12
                is_accident_happen.append([A_class, B_class])
                crash_index.append([i, j])

    return is_accident_happen, crash_index


def counter_vehicles_judge_car_lines(im0, outputs, line_pixel, dividing_pixel, counter_recording, up_counter,
                                     down_counter):
    box_centers = []
    for i, each_box in enumerate(outputs):
        # 求得每个框的中心点
        box_centers.append([(each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2, each_box[4],
                            each_box[5], each_box[2] - each_box[0]])
    # 读取各个框的中心坐标并对上行、下行、总的车辆进行计数，计数原理是将整个视频图像区域划分为四个象限，把第一象限和第三象限的车辆分别认为是下行和上行的车辆
    for box_center in box_centers:
        id_recorded = False
        if len(counter_recording) == 0:
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
        if len(counter_recording) > 0:
            for n in counter_recording:
                if n == box_center[2]:  # 判断该车辆是否已经记过数
                    id_recorded = True
                    break
            if id_recorded:
                continue
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
    # 读取各个框的中心坐标并判断是否进行公交车道区域，并在每辆车上做出标记
    k = []
    b = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    # names = ['car', 'bus', 'truck']
    with open('car_line.txt', 'r') as f:  # 读取前面绘制车道线时保存的文件
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = eval(line)  # 读出来的字符串转化为源格式：字典
            k.append(line['k'])
            b.append(line['b'])
            x1.append(line['x1'])
            y1.append(line['y1'])
            x2.append(line['x2'])
            y2.append(line['y2'])
    # 在视频每一帧中重新绘制禁止驶入区域
    for i in range(len(x1)):
        cv2.line(im0, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 0, 255), 2, cv2.LINE_AA)
    # 判断车辆框的中心是否在绘制的矩形区域内且该车辆是否为公交车,若不是公交车，则违规驶入公交车道。
    # 算法思路为将目标框中心点坐标依次代入四条直线方程内，使用中心点纵坐标分别与代入计算出来的值进行比较，满足所列大小关系即该点在矩形框内部。
    for box_center in box_centers:
        if (k[0] * box_center[0] + b[0] <= box_center[1] and k[1] * box_center[0] + b[1] >= box_center[1]
                and k[2] * box_center[0] + b[2] >= box_center[1] and k[3] * box_center[0] + b[3] <= box_center[1] and (
                        box_center[3] != 1)):
            cv2.putText(im0, 'Entering illegally!', (int(box_center[0] - 140), int(box_center[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, [0, 0, 255], 2)

    if len(outputs) > 0:
        print('s')
    return counter_recording, up_counter, down_counter, box_centers, im0


def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])  # 获得当前帧中跟踪到车辆的ID
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])  # 获得前一帧中跟踪到车辆的ID
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:  # 将当前帧有效检测车辆的信息存入work_locations中
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:  # 将前一帧有效检测车辆的ID索引存入work_IDs_prev_index中
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:  # 将前一帧有效检测车辆的信息存入work_prev_locations中
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +  # 计算有效检测车辆的速度，采用线性的从像素距离到真实空间距离的映射
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *  # 当视频拍摄视角并不垂直于车辆移动轨迹时，测算出来的速度将比实际速度低
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]  # 将保留一位小数的单位为km/h的车辆速度及其ID存入speed二维列表中
    return speed


def draw_speed(img, speed, bbox_xywh, identities):
    for i, j in enumerate(speed):
        for m, n in enumerate(identities):
            if j[1] == n:
                xy = [int(i) for i in bbox_xywh[m]]
                cv2.putText(img, str(j[0]) + 'km/h', (xy[0], xy[1] - 7), cv2.FONT_HERSHEY_PLAIN, 1.5, [255, 255, 255],
                            2)
                break


def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, classes2, identities=None):
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i] * 100))
        label = '%d %s' % (id, cls_names[i])
        # label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def detect(opt, save_img=False):
    # 获取输出文件夹，输入源，权重，参数等参数
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # webcam获取source的信息返回true表示是视频流等文件类型
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # 获得视频的帧宽高
    capture = cv2.VideoCapture(source)
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # deepsort模块初始化
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    # 读取设备
    device = select_device(opt.device)

    # 从训练好的权重文件加载模型
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # 加载数据到dataset里面
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # 从加载好的模型里面读取names模块，为类别信息
    names = model.module.names if hasattr(model, 'module') else model.names
    # class_name = dict(zip(list(range(len(names))), names))

    # 设置计数器
    counter_recording = []
    up_counter = [0] * len(names)  # 设定列表存储不同类别车辆“上行”或“下行”的数目
    down_counter = [0] * len(names)
    # 设置“横向”计数分界线及“纵向”计数分界线，将整个区域分成四部分，用于对“上行”或“下行”车辆进行统计。
    # 若上行和下行车道分界线正好在图像中间，则dividing_pixel = [frame_fature[0] // 2]，
    # 否则将dividing_pixel设置为车道分界线的x轴坐标
    line_pixel = [frame_fature[1] // 2]
    # dividing_pixel = [frame_fature[0] // 2]
    dividing_pixel = [490]
    # 设置每种车型的真实车宽
    width = [1.85, 2.3, 2.5]  # car、bus、truck，单位m
    locations = []
    speed = []

    t0 = time.time()  # 系统时钟的时间戳，ti-t0即为t0到ti之间这段程序运行的系统时间（在程序并发执行时,该时间并非该程序的精确运行时间）
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'  # 设置检测结果保存路径

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 进行推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS(非极大值抑制)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # 判断张量是否为空，即没有检测到目标的情况，为空直接跳过下面的语句，进入下一次循环
            if det is None:
                continue

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            # 碰撞检测
            is_crash, crash_index = find_accidents(det)
            if len(is_crash):
                num = 0
                for i in is_crash:
                    cv2.putText(im0, '%s %s crash' % (names[int(i[0])], names[int(i[1])]), (700, 30 + num),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    num = num + 30  # 隔30个像素显示一个碰撞标签,防止文字重叠
            print(crash_index)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size(用scale_coords函数来将图像缩放)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # 写入结果，绘制目标框
                num = 0
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        colors = [0, 255, 0]
                        if len(crash_index):
                            for i in crash_index:
                                if num in i:
                                    colors = [0, 0, 255]
                        plot_one_box(xyxy, im0, color=colors, line_thickness=3)
                    num += 1

                bbox_xywh = []
                confs = []
                classes = []
                img_h, img_w, _ = im0.shape

                # 把im0的检测结果调整至deepsort的输入数据类型
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                xywhs = torch.Tensor(bbox_xywh)  # 调用Tensor类的构造函数__init__，生成单精度浮点类型的张量
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                # 把调整好的检测结果输入deepsort
                outputs = deepsort.update(xywhs, confss, im0, classes)
                # 计数并判断车辆是否驶入划定的公交车专用区域
                counter_recording, up_counter, down_counter, location, im0 = counter_vehicles_judge_car_lines(im0,
                                                                                                              outputs,
                                                                                                              line_pixel,
                                                                                                              dividing_pixel,
                                                                                                              counter_recording,
                                                                                                              up_counter,
                                                                                                              down_counter)
                # outputs长度为5时传入进行测速。输出location长度也为5，代表五个目标框的信息，其单个元素数据格式：[中心点横坐标、中心点纵坐标、车辆ID、车辆类别、该目标框像素宽度]
                locations.append(location)
                print(len(locations))
                # 每五帧写入一次测速的数据
                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                    with open('speed.txt', 'a+') as speed_record:
                        for sp in speed:
                            speed_record.write('id:%s %skm/h\n' % (str(sp[1]), str(sp[0])))
                    locations = []
                    print('a')
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]
                    draw_speed(im0, speed, bbox_xyxy, identities)
                    draw_boxes(im0, bbox_xyxy, [names[i] for i in classes2], classes2, identities)
                    draw_up_down_counter(im0, up_counter, down_counter, frame_fature, names)
                    # 绘制用于统计车辆的中心横线
                    cv2.line(im0, (0, frame_fature[1] // 2), (frame_fature[0], frame_fature[1] // 2), (0, 0, 100), 2)
                    # cv2.putText(im0, 'Count Dividing Line', (frame_fature[0] // 2 - 100, frame_fature[1] // 2),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 100], 2)
                # 将检测结果写入results.txt中
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
        # 把计数结果写入counter.txt中
        with open('counter.txt', 'w') as counter:
            counter.write('up:%s\ndown:%s' % (str(up_counter), str(down_counter)))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


def get_car_line_and_back_ground(video_path):
    # -------------------------提取背景显示第一帧图像并标定公交专用车道------------------------#
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    imag = cv2.imwrite('background.png', image)
    if imag:
        print('提取背景成功')
    # -------------------------------------专用车道线标定-----------------------------------#
    # 1.读取背景图片
    img = cv2.imread('background.png')
    # cishu是需要标记的点的个数。这个需要标记4点。
    cishu = 4
    sx = []
    im = array(Image.open('background.png'))
    # 2.画实线框点和矩形点
    # 打开交互模式:https://blog.csdn.net/SZuoDao/article/details/52973621
    ion()
    imshow(im)

    for cs in range(cishu):
        print('绘制禁止驶入车道线框须按照顺时针方向，先打左上角第一个点')
        print('Please click 1 points')
        x = ginput(1)
        print('you clicked:', x)
        sx.append(x)
    ioff()
    show()
    print(im.shape)

    # 3.画实线框和矩形框
    jinzhi = []

    # 计算直线方程
    def shixian(x1, y1, x2, y2) -> int:
        if x1 == x2:
            k = -999
            b = 0
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * k
        return k, b

    m = 0
    n = 1
    for i in range(len(sx)):
        if n == 4:
            n = 0
        x = {}
        x1 = int(((sx[m])[0])[0])
        y1 = int(((sx[m])[0])[1])
        x2 = int(((sx[n])[0])[0])
        y2 = int(((sx[n])[0])[1])
        k, b = shixian(x1, y1, x2, y2)
        if y1 > y2:
            yy = y2
            xx = x2
            y2 = y1
            x2 = x1
            y1 = yy
            x1 = xx
        if k != 0:
            for xxx in range(y1, y2):
                xq = (xxx - b) / k
                xq = int(xq)
                cv2.rectangle(img, (int(xq + 2), int(xxx)), (int(xq - 2), int(xxx)), (0, 0, 255), 2)
        else:
            for xxx in range(x1, x2):
                yq = b
                cv2.rectangle(img, (int(xxx), int(yq + 2)), (int(xxx), int(yq - 2)), (0, 0, 255), 2)
        x['k'] = k
        x['b'] = b
        x['x1'] = x1
        x['x2'] = x2
        x['y1'] = y1
        x['y2'] = y2
        print('k:', k, 'b:', b)
        jinzhi.append(x)
        n += 1
        m += 1
    print(jinzhi)
    cv2.imwrite('out.png', img)
    cv2.imshow("label", img)
    cv2.waitKey(0)
    # 将获得的车道线信息写入相应的文件。
    with open("car_line.txt", 'w') as f:
        for s in jinzhi:
            f.write(str(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path')  # 加载的v5权重，可以是公开数据集的预训练权重，也可以是自己数据集训练的权重
    parser.add_argument('--source', type=str, default='test.mp4', help='source')  # 待检测的视频路径
    parser.add_argument('--output', type=str, default='test_out', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2], help='filter by class')  # car、truck、bus
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    # 验证输入图像尺寸是否为32的倍数
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():  # 在该语句下的张量反向传播时不会自动求导，可节省显存或内存
        # 调用该函数绘制禁止驶入区域、获取背景图像，并生成四条直线的各项参数存到car_line.txt里面。
        # 绘制公交专用区域须按照顺时针方向，先打左上角第一个点
        get_car_line_and_back_ground(args.source)
        # 调用该函数进行测速、统计车辆数量、检测碰撞、检测非公交车是否驶入公交车道
        detect(args)
