from styx_msgs.msg import TrafficLight
from obj_det_utils import label_map_util
import os
import numpy as np
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        num_classes = 4
        dir = os.path.dirname(os.path.realpath(__file__))
        path = dir + '/models/frozen_inf_graph_sim_ssd.pb'
        labels_path = dir + '/label_map.pbtxt'

        #Load frozen model
        self.detect_graph = tf.Graph()
        with self.detect_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serial_graph = fid.read()
                graph_def.ParseFromString(serial_graph)
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.detect_graph)

        # We initiate variables to hold bounding boxes, scores and class label of detected objects
        self.boxes = self.detect_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detect_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detect_graph.get_tensor_by_name('detection_classes:0')
        self.image_tensor = self.detect_graph.get_tensor_by_name('image_tensor:0')
        self.num_detections = self.detect_graph.get_tensor_by_name('num_detections:0')

        # Load label mapping
        label_map = label_map_util.load_labelmap(labels_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)

        self.category_index = label_map_util.create_category_index(categories)

    def get_sim_classification(self, image):
        """Determines the color of the traffic light in the image when in simulator
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        detected_light = TrafficLight.UNKNOWN
        img = np.expand_dims(image, axis=0)
        with self.detect_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                          feed_dict={self.image_tensor: img})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if scores[scores.argmax()] > 0.4:
            color = self.category_index[classes[scores.argmax()]]['name']
            if color == 'G':
                detected_light = TrafficLight.GREEN
            elif color == 'R':
                detected_light = TrafficLight.RED
            elif color == 'Y':
                detected_light = TrafficLight.YELLOW
        else:
            detected_light = TrafficLight.UNKNOWN

        return detected_light
    
    def get_site_classification(self, image):
        """Determines the color of the traffic light in the image when in site
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        return TrafficLight.UNKNOWN