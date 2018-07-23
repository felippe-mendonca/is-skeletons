import re
import os
import sys
import json
import cv2
from time import time
from utils.io import ProtobufWriter
from is_wire.core import Logger
from is_msgs.image_pb2 import Image
from google.protobuf.wrappers_pb2 import Int64Value
from skeletons import SkeletonsDetector
from skeletons_utils import load_options, get_np_image, get_pb_image

def get_output_folder(base_folder, dataset):
    output_folder = os.path.join(base_folder, dataset)
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except:
        pass
    return output_folder

op = load_options()
sd = SkeletonsDetector(op)
log = Logger()

if len(sys.argv) != 3:
  log.critical("Please, specify a dataset: python3 detect-video.py options.json <DATASET>")
dataset = sys.argv[2]

with open('video_options.json', 'r') as f:
  vop = json.load(f)
basedir = vop['videos_basedir']
cameras = vop['cameras']
video_file = vop['video_file']
output_dir = get_output_folder(vop['output_dir'], dataset)

for camera in cameras:
  filename = os.path.join(output_dir, 'coco_2d_detector_{camera}'.format(camera=camera))
  writer = ProtobufWriter(filename)

  filename = os.path.join(basedir, dataset, video_file.format(camera=camera))
  vc = cv2.VideoCapture(filename)

  log.info('[{:^10}][{}][{}]', 'Starting', dataset, camera)
  frame_id = 0
  while vc.isOpened:
    ret, frame = vc.read()
    if not ret:
      break
    t0 = time()
    skeletons = sd.detect(frame)
    tf = time()
    dt_ms = (tf - t0)*1000.0
    sequence_id = Int64Value()
    sequence_id.value = frame_id
    writer.insert(sequence_id)
    writer.insert(skeletons)
    log.info('[{:^10}][{}][{}][{}][{:.2f}ms]', 'Detection', dataset, camera, frame_id, dt_ms)

    frame_id += 1
  
  writer.close()