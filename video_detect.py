import cv2
from is_msgs.image_pb2 import Image
import os
from skeletons_pb2 import Skeletons
from skeletons_utils import load_options
from pb_stream import ProtobufWriter
from is_wire.core import Logger
from skeletons import SkeletonsDetector
from threading import Thread
from queue import Queue
from time import time
import sys

op = load_options()
log = Logger()

base_dir = '/felippe/panoptic-dataset-hdvideos'
# datasets = ['160422_haggling1', '160226_haggling1', '160224_haggling1']
datasets = [sys.argv[2]]
video_file = 'hd_{}.mp4'
output_dir = '/felippe/panoptic-dataset-detections'
# cameras = ['00_0{}'.format(i) for i in range(10)]
cameras = [1, 2, 3, 4]

def get_output_folder(base_folder, dataset):
    output_folder = os.path.join(base_folder, dataset)
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except:
        pass
    return output_folder

def worker(q):
    sd = SkeletonsDetector(op)
    while True:
        dataset, camera  = q.get()

        out_dir = get_output_folder(output_dir, dataset)
        output_file = os.path.join(out_dir, 'coco_pose_2d_detected_{}'.format(camera))
        writer = ProtobufWriter(output_file)

        filename = os.path.join(base_dir, dataset, video_file.format(camera))
        vc = cv2.VideoCapture(filename)

        log.info('[{:^10}][{}][{}]', 'Starting', dataset, camera)
        frame_id = 0
        while vc.isOpened:
            ret, frame = vc.read()
            if not ret:
                break

            frame_id += 1
            t0 = time()
            skeletons = sd.detect(frame)
            tf = time()
            dt_ms = (tf - t0)*1000.0
            writer.insert(skeletons)
            log.info('[{:^10}][{}][{}][{}][{:.2f}ms]', 'Detection', dataset, camera, frame_id, dt_ms)

        writer.close()
        vc.release()
        q.task_done()
        log.info('[{:^10}][{}][{}][{} frames]', 'Done!', dataset, camera, frame_id + 1)

n_threads = 1
queue = Queue()
threads = [ Thread(target=worker, args=(queue,)) for _ in range(n_threads)]
for t in threads:
    t.daemon = True
    t.start()

for dataset in datasets:
    for camera in cameras:
        queue.put((dataset, camera))

queue.join()