import sys
from is_wire.core import Logger
from google.protobuf.json_format import Parse
from options_pb2 import Options
from skeletons_pb2 import Skeletons
from is_msgs.image_pb2 import Image
from skeletons_pb2 import SkeletonModel
from skeletons_pb2 import SkeletonPartType as SPT
import cv2
import numpy as np
from itertools import permutations

from time import time

def load_options():
    log = Logger()
    op_file = sys.argv[1] if len(sys.argv) > 1 else 'options.json'
    try:
        with open(op_file, 'r') as f:
            try:
                op = Parse(f.read(), Options())
                log.info('Options: \n{}', op)
                return op
            except Exception as ex:
                log.critical('Unable to load options from \'{}\'. \n{}', op_file, ex)
            except:
                log.critical('Unable to load options from \'{}\'', op_file)
    except Exception as ex:
        log.critical('Unable to open file \'{}\'', op_file)

def get_links(model):
    if model == SkeletonModel.Value('COCO'):
        return [
            (SPT.Value('NECK'), SPT.Value('LEFT_SHOULDER')),
            (SPT.Value('LEFT_SHOULDER'), SPT.Value('LEFT_ELBOW')),
            (SPT.Value('LEFT_ELBOW'), SPT.Value('LEFT_WRIST')),
            (SPT.Value('NECK'), SPT.Value('LEFT_HIP')),
            (SPT.Value('LEFT_HIP'), SPT.Value('LEFT_KNEE')),
            (SPT.Value('LEFT_KNEE'), SPT.Value('LEFT_ANKLE')),
            (SPT.Value('NECK'), SPT.Value('RIGHT_SHOULDER')),
            (SPT.Value('RIGHT_SHOULDER'), SPT.Value('RIGHT_ELBOW')),
            (SPT.Value('RIGHT_ELBOW'), SPT.Value('RIGHT_WRIST')),
            (SPT.Value('NECK'), SPT.Value('RIGHT_HIP')),
            (SPT.Value('RIGHT_HIP'), SPT.Value('RIGHT_KNEE')),
            (SPT.Value('RIGHT_KNEE'), SPT.Value('RIGHT_ANKLE')),
            # (SPT.Value('NECK'), SPT.Value('NOSE')),
            (SPT.Value('NOSE'), SPT.Value('LEFT_EYE')),
            (SPT.Value('LEFT_EYE'), SPT.Value('LEFT_EAR')),
            (SPT.Value('NOSE'), SPT.Value('RIGHT_EYE')),
            (SPT.Value('RIGHT_EYE'), SPT.Value('RIGHT_EAR'))
        ]
    elif model == SkeletonModel.Value('MPI'):
        # TODO
        return []
    else:
        return []

def get_face_parts(model):
    if model == SkeletonModel.Value('COCO'):
        return [
            SPT.Value('NOSE'),
            SPT.Value('LEFT_EYE'),
            SPT.Value('LEFT_EAR'),
            SPT.Value('RIGHT_EYE'),
            SPT.Value('RIGHT_EAR')
        ]
    elif model == SkeletonModel.Value('MPI'):
        # TODO
        return []
    else:
        return []

def get_links_colors():
    return list(permutations([0, 255, 85, 170], 3))

def get_np_image(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


def get_pb_image(input_image, encode_format='.jpeg', compression_level=0.8):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()        
        cimage = cv2.imencode(ext=encode_format, img=input_image, params=params)
        return Image(data=cimage[1].tobytes())
    elif isinstance(input_image, Image):
        return input_image
    else:
        return Image()

def draw_skeletons(input_image, skeletons):
    image = get_np_image(input_image)
    links = get_links(skeletons.model)
    face_parts = get_face_parts(skeletons.model)
    colors = get_links_colors()
    for sk in skeletons.skeletons:
        parts = {}
        for part in sk.parts:
            parts[part.type] = (int(part.x), int(part.y))
        for link_parts, color in zip(links, colors):
            begin, end = link_parts
            if begin in parts and end in parts:
                cv2.line(image, parts[begin], parts[end], color=color, thickness=4)
        for ptype, center in parts.items():
            radius = 2 if ptype in face_parts else 4
            cv2.circle(image, center=center, radius=radius, color=(255, 255, 255), thickness=-1)
    return image