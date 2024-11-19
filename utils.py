import re

import numpy as np

def cal_dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def rotate_point(x, y, angle_degrees=45):
    # 定义坐标点和旋转中心
    point = np.array([x, y])

    # 将坐标点平移到原点
    translated_point = point

    # 构建旋转矩阵
    angle = np.radians(angle_degrees)  # 将角度转换为弧度
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 应用旋转矩阵
    rotated_point = np.dot(rotation_matrix, translated_point)

    # 将坐标点移回原位置
    final_point = rotated_point

    return final_point

def rotate_point_around_center(x, y, cx, cy, angle_degrees=45):
    # 定义坐标点和旋转中心
    point = np.array([x, y])
    center = np.array([cx, cy])

    # 将坐标点平移到原点
    translated_point = point - center

    # 构建旋转矩阵
    angle = np.radians(angle_degrees)  # 将角度转换为弧度
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 应用旋转矩阵
    rotated_point = np.dot(rotation_matrix, translated_point)

    # 将坐标点移回原位置
    final_point = rotated_point + center

    return final_point

def iou(box1, box2):
    # 计算两个矩形的交集区域的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集区域的面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个矩形的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IOU
    iou = intersection_area / union_area
    return iou

def get_center(bbox):
    return (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2

def load_plan(path):
    with open(path, 'r', encoding='utf8') as f:
        return eval(f.read())


def match_any_pattern(string, patterns):
    """
    Checks if a given string matches any pattern in a list of patterns.

    Args:
        string (str): The string to be checked.
        patterns (list): A list of regular expression patterns.

    Returns:
        bool: True if the string matches any pattern, False otherwise.
    """
    for pattern in patterns:
        if re.match(pattern, string):
            return True
    return False

def psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))