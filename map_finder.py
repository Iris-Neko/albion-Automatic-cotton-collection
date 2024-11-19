import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm


class SmallMapFinder:
    def __init__(self):
        self.match_scales = np.exp(np.linspace(np.log(0.5), np.log(2.0), 21))
        self.last_scale = None
        self.small_map_center = (1712, 910)
        self.small_map_size = (141, 100)

    def template_matching(self, image, template, scale=1.0):
        if scale != 1:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized_template = template

        # Perform template matching
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_val, max_loc

    def multi_scale_template_matching(self, template, image):
        """
        Perform multi-scale template matching between the template and the image.

        :param template: Template image (smaller image).
        :param image: Larger image in which to find the template.
        :param scales: Array of scale factors to apply to the template.
        :return: Top-left corner and scale of the best match.
        """
        template_height, template_width = template.shape[:2]
        best_match = None
        best_scale = 1
        max_corr = 0

        if self.last_scale is not None:
            max_val, max_loc = self.template_matching(image, template, self.last_scale)

            # Update the best match if the current state is better
            if max_val > max_corr:
                max_corr = max_val
                best_match = max_loc
                best_scale = self.last_scale

        if max_corr <= 0.5:
            # Loop over the scales
            for scale in self.match_scales:
                # Resize the template according to the current scale
                max_val, max_loc = self.template_matching(image, template, scale)

                # Update the best match if the current state is better
                if max_val > max_corr:
                    max_corr = max_val
                    best_match = max_loc
                    best_scale = scale

        if best_match is not None:
            self.last_scale = best_scale
            #print(f"Best match found at: {best_match} with scale: {best_scale} and correlation: {max_corr}")
            return best_match, best_scale
        else:
            print("No match found.")
            return None

    def extract_diamond(self, image):
        """根据中心点和半径从图像中切出菱形区域"""
        x, y = self.small_map_center
        w, h = self.small_map_size

        image = image[y - h:y + h, x - w:x + w, :]

        mask = np.zeros_like(image)
        points = np.array([
            [w, h - h],  # 上顶点
            [w + w, h],  # 右顶点
            [w, h + h],  # 下顶点
            [w - w, h]  # 左顶点
        ])
        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        diamond = cv2.bitwise_and(image, mask)
        # diamond = diamond[y - h: y + h, x - w: x + w]
        resized_map = cv2.resize(diamond, (w, w))
        return resized_map

    def rotate_image(self, image, angle=45):
        """将图像逆时针旋转指定角度并裁剪去除黑边"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_val = np.abs(matrix[0, 0])
        sin_val = np.abs(matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        rotated = cv2.warpAffine(image, matrix, (new_width, new_height))
        # 计算裁剪尺寸
        crop_size = 100
        start = round((new_width - crop_size) / 2)
        return rotated[start:start + crop_size, start:start + crop_size]

    def find_one_image(self, small_map, large_map, show=False):
        # 切出菱形区域
        diamond = self.extract_diamond(small_map)
        # 逆时针旋转45度
        rotated = self.rotate_image(diamond, 45)
        small_map = rotated

        if show:
            # 显示结果
            cv2.imshow('small_map', small_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        image_gray = cv2.cvtColor(large_map, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(small_map, cv2.COLOR_BGR2GRAY)

        # Perform multi-scale template matching
        best_location, best_scale = self.multi_scale_template_matching(template_gray, image_gray)
        return best_location, best_scale, template_gray

    def find_one(self, small_map_path='map.png', large_map_path='maps/3207.png'):
        # 加载图像
        small_map = cv2.imread(small_map_path)
        large_map = cv2.imread(large_map_path)

        best_location, best_scale, template_gray = self.find_one_image(small_map, large_map, show=True)

        # Optionally, show the result on the image
        if best_location:
            top_left = best_location
            h, w = template_gray.shape[:2]
            bottom_right = (top_left[0] + int(w * best_scale), top_left[1] + int(h * best_scale))
            cv2.rectangle(large_map, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imshow('Match', large_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def process_video(self, large_map_path, video_input_path, video_output_path):
        # 加载大图
        large_map_ = cv2.imread(large_map_path)
        #large_map_ = cv2.resize(large_map_, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # 准备视频读取和写入
        cap = cv2.VideoCapture(video_input_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        frames_num = cap.get(7)
        pbar = tqdm(total=frames_num)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                pbar.update()

                large_map = deepcopy(large_map_)
                best_location, best_scale, template_gray = self.find_one_image(frame, large_map)

                if best_location:
                    top_left = best_location
                    h, w = template_gray.shape[:2]
                    bottom_right = (top_left[0] + int(w * best_scale), top_left[1] + int(h * best_scale))
                    cv2.rectangle(large_map, top_left, bottom_right, (0, 255, 0), 2)

                # 显示结果
                cv2.imshow('Match', large_map)

                # 写入输出视频
                out.write(large_map)

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    finder = SmallMapFinder()
    finder.process_video('maps/3207.png', 'datas/test.mp4', 'output.mp4')
