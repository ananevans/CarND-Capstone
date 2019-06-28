#!/usr/bin/env python
# Print total cumulative serialized msg size per topic
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2 as cv
import numpy as np
import argparse

bridge = CvBridge()
    
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
        # im = np.array([msg.data.data])
        # resized = cv2.resize(im.data, (im.width, im.height))
        cv.imwrite('imgs/' +str(i) +'.png', im)
        i+=1
        if (i%100 == 0):
            print i
    bag.close()

if __name__ == '__main__':
    main()

