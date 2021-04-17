# tflite_object_detection_opencv

* The model file and most of the code are taken from the repo [tensorflow examples](https://github.com/tensorflow/examples), and to be explicit from [this](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi) folder

* I modified the code to work with opencv instaed of pillow libary and changed the annotation part for my own purposes.

```
python3 tflite_inference.py -m ./coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite -l ./coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/coco_labels.json
```