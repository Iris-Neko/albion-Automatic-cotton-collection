import cv2
import numpy as np


class MapTransformer:
    def __init__(self, center=(540, 960), size=(249, 198), r=100):
        self.pts_from = np.float32([
            [center[0] - size[0] / 2, center[1]],  # 椭圆左侧点
            [center[0] + size[0] / 2, center[1]],  # 椭圆右侧点
            [center[0], center[1] - size[1] / 2],  # 椭圆顶部点
            [center[0], center[1] + size[1] / 2]  # 椭圆底部点
        ])

        self.pts_to = np.float32([
            [center[0] - r, center[1]],  # 原始圆左侧点
            [center[0] + r, center[1]],  # 原始圆右侧点
            [center[0], center[1] - r],  # 原始圆顶部点
            [center[0], center[1] + r]  # 原始圆底部点
        ])

        self.transM = cv2.getPerspectiveTransform(self.pts_from, self.pts_to)

    def __call__(self, img):
        return cv2.warpPerspective(img, self.transM, (img.shape[1], img.shape[0]))

trans = MapTransformer(center=(320,240))

# 假设我们有一个变换后的图像img
img = cv2.imread('transformed_image.png')

# 应用透视变换
dst = trans(img)

# 显示图像
cv2.imshow('Original Image', img)
cv2.imshow('Perspective Transformation', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
