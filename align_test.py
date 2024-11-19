import cv2
from capture import WindowCapture
import numpy as np
import time

if __name__ == '__main__':
    time.sleep(2)

    capture = WindowCapture("Albion Online Client")
    pc = np.array([960, 475])
    pc2 = 1000

    pts1 = np.float32([[600+pc[0], 400+pc[1]], [-600+pc[0], -400+pc[1]], [-600+pc[0], 400+pc[1]], [600+pc[0], -400+pc[1]]])  # 原始图像四个角点坐标
    pts2 = np.float32([[400+pc[0], -100+pc2], [-600+pc[0], 100+pc2], [0+pc[0], 420+pc2], [0+pc[0], -660+pc2]])  # 目标图像四个角点坐标

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_game = capture.capture()

    # 进行透视变换
    dst = cv2.warpPerspective(img_game, M, (2500, 2000))

    rows, cols = dst.shape[:2]
    M = cv2.getRotationMatrix2D((960, 540), -50, 1)
    rotated = cv2.warpAffine(dst, M, (cols, rows))

    # 显示结果
    cv2.imshow('original', img_game)
    cv2.imshow('perspective', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()