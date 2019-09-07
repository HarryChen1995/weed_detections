import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import Counter
from io import StringIO
from matplotlib import pyplot as plt 
plt.switch_backend('Agg')
from PIL import Image

import cv2

sys.path.append("..")



from utils import label_map_util

from utils import visualization_utils as vis_util

def predict(image_path):
    MODEL_NAME = 'weed'

    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
  
    PATH_TO_LABELS = os.path.join(MODEL_NAME, 'weed_label.pbtxt')

    NUM_CLASSES = 3

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')




    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_np=cv2.imread(image_path)
    
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

    result = zip(np.squeeze(scores),np.squeeze(classes))
    result = [i for i in result if i[0] >0.5]
    max_result = Counter()
    for Score, Class in result:
        if max_result[str(int(Class))] < Score:
            max_result[str(int(Class))]=Score
    max_result = dict(max_result)
    return image_np,max_result