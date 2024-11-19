from capture import WindowCapture
from map_finder import SmallMapFinder
import time
import cv2
from controller import MouseController
import numpy as np
import math


def player_pos(map_finder, img_game, large_map):
    top_left, best_scale, template_gray = map_finder.find_one_image(img_game, large_map)
    h, w = template_gray.shape[:2]
    return top_left[0] + int(w * best_scale) // 2, top_left[1] + int(h * best_scale) // 2

def get_offset(px, py):
    img_game = capture.capture()
    pos1 = player_pos(finder, img_game, large_map)

    dist = math.sqrt(px ** 2 + (py*1.414) ** 2)
    t = 2.5 * dist / 900 + 0.5

    mouse_ctrl.click((int(player_center[0]+px), int(player_center[1]+py)))
    time.sleep(t)

    img_game = capture.capture()
    pos2 = player_pos(finder, img_game, large_map)

    return pos2[0]-pos1[0], pos2[1]-pos1[1]

if __name__ == '__main__':
    time.sleep(2)

    capture = WindowCapture("Albion Online Client")
    finder = SmallMapFinder()
    large_map = cv2.imread(f'maps/{4213}.png')

    mouse_ctrl = MouseController(capture.window.left + 8, capture.window.top + 31)
    #center = np.array([960, 540])
    player_center = np.array([960, 475])

    time.sleep(0.5)

    pos_list = []

    def get_one(px, py):
        pos = get_offset(px, py)
        print(px, py, pos)
        pos_list.append([px,py,*pos])

    get_one(600, 400)
    get_one(-600, -400)
    get_one(-600, 400)
    get_one(600, -400)

    # for py in range(100, 450 + 1, 150):
    #     for px in range(100, 900+1, 200):
    #         pos = get_offset(px, py)
    #         print(px, py, pos)
    #         pos = get_offset(-px, -py)
    #         print(-px, -py, pos)
    #         pos = get_offset(-px, py)
    #         print(-px, py, pos)
    #         pos = get_offset(px, -py)
    #         print(px, -py, pos)

    radius = 2

    map_area = large_map[700:900,700:900]

    # 画点
    for pos in pos_list:
        cv2.circle(map_area, (100+pos[2], 100+pos[3]), radius, (255, 0, 0), -1)

    # 显示图像
    cv2.imshow('Image', map_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    600 400 (19, -3)
    -600 -400 (-35, 7)
    -600 400 (0, 21)
    600 -400 (-1, -33)
    
    600 400 (19, -3)
    -600 -400 (-29, 5)
    -600 400 (-4, 23)
    600 -400 (5, -35)
    
    600 400 (20, -3)
    -600 -400 (-30, 5)
    -600 400 (0, 20)
    600 -400 (0, -32)
    
    600 400 (20, -4)
    -600 -400 (-30, 5)
    -600 400 (0, 21)
    600 -400 (0, -33)
    '''

    '''
    600 400 (20, -5)
    -600 -400 (-30, 5)
    -600 400 (0, 21)
    600 -400 (0, -33)
    '''
