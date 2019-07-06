#!/usr/bin/env python
# Print total cumulative serialized msg size per topic
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2 as cv
import numpy as np
import argparse
import tl_classifier as tlc

bridge = CvBridge()
classifier = tlc.TLClassifier(1)
    
def main():
    parser = argparse.ArgumentParser(description='Analyse Path Following')
    parser.add_argument('--BagFile', help='an integer for the accumulator')

    args = parser.parse_args()
    bag_file = args.BagFile

    # open bag
    bag  = rosbag.Bag(bag_file)
    i = 0
    # loop over the topic to read evey message
    for topic, msg, t in bag.read_messages(topics='/image_raw'):
        im = bridge.imgmsg_to_cv2(msg, 'bgr8')
        color = classifier.get_classification(im)
        cv.imwrite(color+'/' +str(i) +'.png', im)
        i+=1
        if (True):
            print i
    bag.close()

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    main()

