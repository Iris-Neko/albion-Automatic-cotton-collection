from ultralytics import YOLO
from capture import WindowCapture
import time

if __name__ == '__main__':
    time.sleep(2)

    # Load a model
    model = YOLO('ckpts/best.pt')  # pretrained YOLOv8n model
    capture = WindowCapture("Albion Online Client")

    names = model.names

    for i in range(10):
        img_game = capture.capture()

        # Run batched inference on a list of images
        results = model(img_game)  # return a list of Results objects

        # Process results list
        for result in results:
            print('pred')
            for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
                print(names[int(cls)], conf, xyxy)
            time.sleep(1)