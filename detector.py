from ultralytics import YOLO
from utils import iou, get_center, cal_dist, match_any_pattern
import numpy as np
import cv2
from collections import Counter

class Detector:
    def __init__(self, center=(960-5, 540-70), max_area=200*200):
        self.yolo = YOLO('ckpts/best_v1.pt')
        self.cls_names = self.yolo.names
        self.center = center
        self.max_area = max_area

        self.det_list=[]

    def detect(self, img_game, cls_accept='all', offset=(0,0)):
        img_game[self.center[1]-20-50:self.center[1]-20+50, self.center[0]-40:self.center[0]+40, :]=0
        img_game[1712-141:1712+141, 910-100:910+100, :]=0

        h, w = img_game.shape[:2]
        sw, sh = w/640, h/640
        resized_img_game = cv2.resize(img_game, (640, 640), interpolation=cv2.INTER_AREA)

        results = self.yolo(resized_img_game, imgsz=[640, 640], conf=0.2)
        obj_list = []
        for result in results:
            for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
                cls_name = self.cls_names[int(cls)]
                if cls_accept!='all' and not match_any_pattern(cls_name, cls_accept):
                    continue
                xyxy = xyxy.tolist()
                bbox = [xyxy[0]*sw+offset[0], xyxy[1]*sh+offset[1], xyxy[2]*sw+offset[0], xyxy[3]*sh+offset[1]]
                if (bbox[2]-bbox[0])*(bbox[3]-bbox[1])>self.max_area:
                    continue
                obj_list.append((cls_name, conf.item(), bbox))
        self.det_list.append(obj_list)

    def get_score(self, group):
        score_list = []
        for item in group:
            cls_count = Counter([x[0] for x in item])
            cls_count = max(cls_count.values())
            dist = cal_dist(*get_center(item[0][2]), *self.center)
            score_list.append(cls_count-dist/100)
        return score_list

    def get_res(self):
        bbox_group = [[item] for item in self.det_list[0]]

        for det in self.det_list[1:]:
            for det_item in det:
                for item in bbox_group:
                    if iou(det_item[2], item[0][2])>0.75:
                        item.append(det_item)
                        break
                else:
                    bbox_group.append([det_item])

        score_list = self.get_score(bbox_group)
        if len(score_list)==0:
            return None
        max_id = np.argmax(score_list)
        bbox = np.array([item[2] for item in bbox_group[max_id]]).mean(axis=0)

        self.det_list.clear()
        return bbox

if __name__ == '__main__':
    import time
    from capture import WindowCapture

    time.sleep(2)

    detector = Detector()

    capture = WindowCapture("Albion Online Client")
    img = capture.capture()
    detector.detect(img, 'all')

    bbox = detector.get_res()

    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    # 显示带有边界框的图像
    cv2.imshow('image with bounding box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()