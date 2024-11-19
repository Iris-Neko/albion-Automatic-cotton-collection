import time

import cv2
from retrying import retry

from capture import WindowCapture
from utils import psnr


class ObjChecker:
    def __init__(self, capture: WindowCapture, center, res_bar_path='imgs/res_bar.png', bull_path='imgs/bull.png'):
        self.capture = capture
        self.center = center
        self.res_bar = cv2.imread(res_bar_path)
        self.res_area = (856, 675, 856 + 15, 675 + 23)

        self.bull = cv2.imread(bull_path)

    def check_pick_finish(self):
        img_game = self.capture.capture()
        img_corp = img_game[self.res_area[1]:self.res_area[3], self.res_area[0]:self.res_area[2], :]
        result = psnr(img_corp, self.res_bar)
        return result

    def wait_pick_finish(self, thr=15, miss_thr=2, t=0.1):
        miss_count = 0

        while True:
            max_score = self.check_pick_finish()
            if max_score < thr:
                miss_count += 1
            else:
                miss_count = 0

            if miss_count >= miss_thr:
                break
            time.sleep(t)

    @retry(stop_max_attempt_number=5, wait_fixed=1000)
    def check_bull(self, thr=20):
        cursor_icon = self.capture.get_cursor_icon()
        c_psnr = psnr(self.bull, cursor_icon[:, :, :3])
        return c_psnr > thr


if __name__ == '__main__':
    capture = WindowCapture("Albion Online Client")
    checker = ObjChecker(capture, (960, 540))

    time.sleep(3)

    checker.wait_pick_finish(1)
    # checker.check_bull()
    print('finish')
