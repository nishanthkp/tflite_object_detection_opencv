# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

from utils import Color, draw_text_with_background


def load_labels(path: str) -> dict:
    labels = dict()

    with open(path, 'r', encoding='utf-8') as label_file:
        labels = json.load(label_file)
    return labels


def annotate(image, results, labels):
    width = image.shape[1]
    height = image.shape[0]
    for result in results:
        ymin, xmin, ymax, xmax = result['bounding_box']
        xmin = int(xmin * width)
        ymin = int(ymin * height)
        xmax = int(xmax * width)
        ymax = int(ymax * height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), Color['LIME'])
        draw_text_with_background(image, labels[str(result['class_id'])], (xmin, ymin))


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = len(classes)

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': int(scores[i] * 100)
            }
            results.append(result)
    return results


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', required=True)
    arg_parser.add_argument('-l', '--labelmap', required=True)
    arg_parser.add_argument('-v', '--video', default=0)
    arg_parser.add_argument('-t', '--threshold', default=0.4)
    arg_parser.add_argument('-o', '--output', default='output.avi')
    args = arg_parser.parse_args()

    labels = load_labels(args.labelmap)

    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    stream = cv2.VideoCapture(args.video)
    if not stream.isOpened():
        logging.critical('unable to open input stream')
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while stream.isOpened():
        ok, original_frame = stream.read()
        if not ok:
            logging.critical('cannot read frame')
            exit()

        frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (input_width, input_height))

        results = detect_objects(interpreter, frame, args.threshold)
        annotate(original_frame, results, labels)

        out.write(original_frame)
        cv2.imshow('stream', original_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.release()
    out.release()


if __name__ == '__main__':
    main()
