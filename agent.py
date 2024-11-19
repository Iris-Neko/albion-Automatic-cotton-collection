import time

import sys
import numpy as np
import random
import math
import argparse

from map_finder import SmallMapFinder
import cv2
from typing import List
from capture import WindowCapture
from controller import MouseController, MOUSE_LEFT
from utils import rotate_point, cal_dist
from detector import Detector
from obj_checker import ObjChecker

flag = False

plan1=[
    ('goto', 800, 600),
    ('pick', 're:^fiber.*'),
    ('goto', 500, 600),
    ('put', 'all')
]

import keyboard

def on_key_event(e):
    if e.name == 'p':
        print("Exiting program...")
        keyboard.unhook_all()  # 停止监听键盘事件
        global flag
        flag = True
        sys.exit(0)

keyboard.on_press(on_key_event)

class AlbionAgent:
    def __init__(self, map_id:int):
        self.capture = WindowCapture("Albion Online Client")
        self.large_map = cv2.imread(f'maps/{map_id}.png')
        self.map_finder = SmallMapFinder()
        self.mouse_ctrl = MouseController(self.capture.window.left+8, self.capture.window.top+31+15)
        self.detector = Detector()

        self.center = np.array([960, 540])
        self.obj_checker = ObjChecker(self.capture, self.center)

        self.capture_new()

        self.last_pick_pos = None

    def capture_new(self):
        self.img_game = self.capture.capture()

    @property
    def player_pos(self):
        top_left, best_scale, template_gray = self.map_finder.find_one_image(self.img_game, self.large_map)
        h, w = template_gray.shape[:2]
        return top_left[0] + int(w * best_scale)//2, top_left[1] + int(h * best_scale)//2

    def move_one_step(self, x, y, step=400, deg=45, t=0.5):
        player_pos = self.player_pos
        vx, vy = x - player_pos[0], y - player_pos[1]
        dist = np.sqrt(vx ** 2 + vy ** 2)
        if dist<0.2:
            return player_pos
        vx /= dist
        vy /= dist

        pos = rotate_point(vx * step, vy * step, deg)
        # pos[1]=-pos[1]
        pos += self.center
        self.mouse_ctrl.click((int(pos[0]), int(pos[1])))
        time.sleep(t)
        self.capture_new()
        return player_pos

    def move_to(self, x, y, step=400):
        global flag
        player_pos = self.player_pos
        last_player_pos = (0,0)

        while True:
            # 如果按下 'p' 键，则退出循环
            if flag:
                sys.exit(0)
                return

            player_pos = self.move_one_step(x, y, step, deg=45+random.uniform(-5,5))

            if cal_dist(*player_pos, x, y)<8:
                break

            add_deg = 45
            rot_dir = random.randint(0,1)*2-1
            print(player_pos, x,y)

            while cal_dist(*player_pos, *last_player_pos)<3: # 卡住了
                for i_try in range(2):
                    player_pos = self.move_one_step(x, y, step, deg=rot_dir*add_deg, t=0.8)
                    if cal_dist(*player_pos, *self.player_pos)<2:
                        break
                    last_player_pos = player_pos
                else:
                    break

                add_deg+=45
                # 如果按下 'p' 键，则退出循环
                if flag:
                    sys.exit(0)
                    return
                #last_player_pos = player_pos

            last_player_pos = player_pos

    def rid_bull(self, r=200):
        sr = math.sqrt(r)
        cx, cy = self.center[0] + 40, self.center[1] - 60

        pos_list=[(cx, cy-r), (cx-sr, cy-sr), (cx-r, cy), (cx-sr, cy+sr), (cx, cy+r), (cx+sr, cy+sr), (cx+r, cy), (cx+sr, cy-sr)]

        for px, py in pos_list:
            self.mouse_ctrl.to(int(px), int(py))
            time.sleep(0.05)
            if self.obj_checker.check_bull():
                self.mouse_ctrl.click((int(px), int(py)), button=MOUSE_LEFT)
                time.sleep(0.5)
                break


    def move_det(self, cls_accept, cx, cy, dx, dy, t=0.1, offset=(0, 0)):
        self.mouse_ctrl.click((int(cx), int(cy)))
        time.sleep(t)
        for i in range(2):
            self.capture_new()
            # 因为移动会导致检测框偏移，需要一些修正
            self.detector.detect(self.img_game, cls_accept, offset=((offset[0]), (offset[1])))

    def _pos2time(self, pos):
        x, y = pos[0]-self.center[0], (pos[1]-self.center[1])*1.414
        dist = math.sqrt(x**2+y**2)
        t = 2.5*dist/900 + 0.5
        return t

    def pick_up(self, cls_accept, has_bull=True, show=True):
        cx, cy = self.center[0]-5, self.center[1]-70

        # 如果按下 'p' 键，则退出
        if flag:
            sys.exit(0)
            return

        #pos_list=[(30,0), (0,30), (-30,0), (0, -30)]
        pos_list=[(30,0), (-30, 0)]

        self.capture_new()
        self.detector.detect(self.img_game, cls_accept)

        # ofx, ofy=0, 0
        # for px, py in pos_list:
        #     ofx+=px+16
        #     ofy+=py
        #     self.move_det(cls_accept, cx+px, cy+py, px, py, offset=(ofx, ofy))


        bbox = self.detector.get_res()
        if bbox is None:
            print('find nothing')
            return False

        if show:
            self.capture_new()
            cv2.rectangle(self.img_game, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            # 显示带有边界框的图像
            cv2.imshow('image with bounding box', self.img_game)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        tx = (bbox[0]+bbox[2])/2
        ty = (bbox[1]+bbox[3])/2
        #self.mouse_ctrl.move_to(int(cx), int(cy))
        t_pos = (int(tx), int(ty))
        self.mouse_ctrl.click(t_pos, button=MOUSE_LEFT, t=0.3)

        # wait for pick up finish
        time.sleep(self._pos2time(t_pos))
        self.obj_checker.wait_pick_finish()

        if has_bull:
            self.rid_bull(100)

        pos_after = self.player_pos
        if self.last_pick_pos and cal_dist(*self.last_pick_pos, *pos_after) < 2:
            self.mouse_ctrl.click((int(cx+random.uniform(-300, 300)), int(cy+random.uniform(-300, 300))))
            time.sleep(0.6)
        self.last_pick_pos = pos_after

        return True

    def pick_up_loop(self, cls_accept, has_bull=True, show=False):
        while self.pick_up(cls_accept, show=show):
            pass

    def zoom_map(self):
        self.mouse_ctrl.to(*self.map_finder.small_map_center)
        time.sleep(0.1)
        self.mouse_ctrl.scroll(20)
        time.sleep(0.5)

    # def pick_up_loop(self, cls_accept, has_bull=True, show=False):
    #     start = []
    #
    #     while self.pick_up(cls_accept, show=show):
    #         if has_bull:
    #             px, py = self.player_pos
    #             if len(start)==0:
    #                 start.append(px)
    #                 start.append(py)
    #             else:
    #                 if cal_dist(px, py, *start)>8:
    #                     self.rid_bull()
    #                     start.clear()
    #
    #     self.rid_bull()


    def run_action(self, action):
        if action[0] == 'goto':
            self.move_to(int(action[1]), int(action[2]))
        elif action[0] == 'pick':
            self.pick_up(action[1])
        elif action[0] == 'pick_loop':
            self.pick_up_loop(action[1])
        elif action[0] == 'sleep':
            time.sleep(action[1])

    def run_plan(self, plan:List):
        for action in plan:
            print(action)
            self.run_action(action)

    def run_plan_loop(self, plan:List, to_nearest=True):
        self.zoom_map()

        plan_idx = 0
        pos_arr = np.array([(action[1], action[2]) if action[0]=='goto' else (10000, 10000) for action in plan])

        self.capture_new()
        pos_now = self.player_pos
        distances = np.linalg.norm(pos_arr - np.array(pos_now), axis=1)
        plan_idx = distances.argmin()

        while True:
            action = plan[plan_idx]
            self.run_action(action)
            if to_nearest and action[0]=='pick_loop':
                self.capture_new()
                pos_now = self.player_pos
                distances = np.linalg.norm(pos_arr - np.array(pos_now), axis=1)
                distances[:plan_idx]=10000
                distances[plan_idx+10:]=10000
                plan_idx = distances.argmin()
                print(plan_idx)
                print(f'switch to: {plan[plan_idx]}')
            else:
                plan_idx+=1

            if plan_idx>=len(plan):
                plan_idx=0


if __name__ == '__main__':
    from utils import load_plan

    parser = argparse.ArgumentParser(description='agent argument')
    parser.add_argument('map_id', type=str, default='0209')
    parser.add_argument('--plan', type=str, default='plan_cotton')
    parser.add_argument('--loop', action='store_true', default=False)
    args = parser.parse_args()

    # plan = [
    #     ('pick_loop', ['.*']), # '^fiber.*t[^2]$'
    #     #('pick', 'all'),
    # ]

    plan = load_plan(f'{args.plan}.txt')

    time.sleep(2)
    agent = AlbionAgent(args.map_id)
    if args.loop:
        agent.run_plan_loop(plan)
    else:
        agent.run_plan(plan)

    # plan2 = load_plan('plan1.txt')
    #
    # time.sleep(2)
    # agent = AlbionAgent('3206')
    # agent.run_plan_loop(plan2)