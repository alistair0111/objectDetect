# import numpy as np
# import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import pyttsx3
# import zipfile

# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image

# import tensorflow as tf

# from utils import label_map_util

# from utils import visualization_utils as vis_util

# import cv2
# import speech_recognition as sr

# r = sr.Recognizer()

# cap = cv2.VideoCapture(
#     0
# )  # if you have multiple webcams change the value to the correct one

# # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# #
# # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# # In[4]:

# # What model to download.
# MODEL_NAME = "ssd_mobilenet_v1_coco_11_06_2017"
# MODEL_FILE = MODEL_NAME + ".tar.gz"
# DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"

# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join("data", "mscoco_label_map.pbtxt")

# NUM_CLASSES = 90

# def t2s(command):
#     # Initialize the engine
#     engine = pyttsx3.init()
#     rate = engine.getProperty("rate")
#     en_voice_f = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
#     engine.setProperty("voice", en_voice_f)
#     engine.setProperty("rate", rate)
#     engine.say(command)
#     engine.runAndWait()

# # ## Download Model

# # In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if "frozen_inference_graph.pb" in file_name:
#         tar_file.extract(file, os.getcwd())

# # ## Load a (frozen) Tensorflow model into memory.

# # In[6]:

# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.compat.v1.GraphDef()
#     with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name="")

# # ## Loading label mapQQ
# # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# # In[7]:

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True
# )
# category_index = label_map_util.create_category_index(categories)

# # ## Helper code

# # In[8]:

# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# # # Detection

# # In[9]:

# # For the sake of simplicity we will use only 2 images:
# # image1.jpg
# # image2.jpg
# # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = "test_images"
# TEST_IMAGE_PATHS = [
#     os.path.join(PATH_TO_TEST_IMAGES_DIR, "image{}.jpg".format(i)) for i in range(1, 3)
# ]  # change this value if you want to add more pictures to test

# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

# # In[10]:

# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         while True:
#             ret, image_np = cap.read()
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
#             # Each box represents a part of the image where a particular object was detected.
#             boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
#             # Each score represent how level of confidence for each of the objects.
#             # Score is shown on the result image, together with the class label.
#             scores = detection_graph.get_tensor_by_name("detection_scores:0")
#             classes = detection_graph.get_tensor_by_name("detection_classes:0")
#             num_detections = detection_graph.get_tensor_by_name("num_detections:0")
#             # Actual detection.
#             (boxes, scores, classes, num_detections) = sess.run(
#                 [boxes, scores, classes, num_detections],
#                 feed_dict={image_tensor: image_np_expanded},
#             )
#             # Visualization of the results of a detection.
#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image_np,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores),
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=3,
#             )
#             my_list = [
#                 category_index.get(value)
#                 for index, value in enumerate(classes[0])
#                 if scores[0, index] > 0.5
#             ]
#             res = {}
#             for line in my_list:
#                 res.update(line)
#                 t2s("I can See a" + res["name"] + "is here")

#             cv2.imshow("object detection", cv2.resize(image_np, (800, 600)))

#             if cv2.waitKey(25) & 0xFF == ord("q"):
#                 cv2.destroyAllWindows()
#                 break

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import pyttsx3
import zipfile
import streamlit as st

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import io

from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2
import speech_recognition as sr

r = sr.Recognizer()

cap = cv2.VideoCapture(
    0)  # if you have multiple webcams change the value to the correct one

# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = "ssd_mobilenet_v1_coco_11_06_2017"
MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("data", "mscoco_label_map.pbtxt")

NUM_CLASSES = 90


def t2s(command):
    # Initialize the engine
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    en_voice_f = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty("voice", en_voice_f)
    engine.setProperty("rate", rate)
    engine.say(command)
    engine.runAndWait()


# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if "frozen_inference_graph.pb" in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

# ## Loading label mapQQ
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = "test_images"
TEST_IMAGE_PATHS = [
    os.path.join(PATH_TO_TEST_IMAGES_DIR, "image{}.jpg".format(i))
    for i in range(1, 3)
]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# In[10]:
st.title("Webcam Live Feed")
run = st.checkbox("Run", key='checkbox')
FRAME_WINDOW = st.image([])

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name(
                "num_detections:0")
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
            )
            my_list = [
                category_index.get(value)
                for index, value in enumerate(classes[0])
                if scores[0, index] > 0.5
            ]
            res = {}
            for line in my_list:
                res.update(line)
                # print("Res", res)
                t2s("I can see a" + res["name"] + "is here")

            # camera = cv2.VideoCapture(0)

            # while run:
            # _, frame = camera.read()
            # frame = cv2.imshow("object detection",
            #                    cv2.resize(image_np, (800, 600)))
            # dataBytesIO = io.BytesIO(frame)
            FRAME_WINDOW.image(image_np)
            # else:
            #     st.write("Stopped")

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
