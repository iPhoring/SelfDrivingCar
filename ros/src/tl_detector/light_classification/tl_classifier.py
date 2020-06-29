from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import tensorflow as tf


class TLClassifier(object):
    def __init__(self,  modelpath):
        self.threshold = 0.3
        inference_path = modelpath
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(inference_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.graph, config=config)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args: image (cv::Mat): image containing the traffic light
        Returns: int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dim = image.shape[0:2]

        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)

            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            rospy.loginfo("[TL_Classifier] Score:[{}]".format(max(scores)))

            for box, score, class_label in zip(boxes, scores, classes):
                if score > self.threshold:
                    class_label = int(class_label)
                    if class_label == 2:
                        rospy.loginfo("[TL_Classifier] {RED}")
                        return TrafficLight.RED
                    elif class_label == 3:
                        rospy.loginfo("[TL_Classifier] {YELLOW}")
                        return TrafficLight.YELLOW
                    elif class_label == 1:
                        rospy.loginfo("[TL_Classifier] {GREEN}")
                        return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
