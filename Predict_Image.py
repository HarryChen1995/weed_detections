import numpy as np
import tensorflow as tf
from collections import Counter
from matplotlib import pyplot as plt 
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

#wrapper function 
def predict(image_path):

    inference_graph_path = 'weed/frozen_inference_graph.pb'
    weed_label_path = 'data/weed_label.pbtxt'
    

    #import trained model from protocol buffer 
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as f:
            serialized_graph = f.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')



    #read weed label map file and map labels to interger value 
    label_map = label_map_util.load_labelmap(weed_label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=3, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # run inferences on trained model
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_np=cv2.imread(image_path)
    
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
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
    
    #send back recognized images and accuracy of prediction back to GUI
    return image_np,max_result