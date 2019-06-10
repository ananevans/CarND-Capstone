#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 9

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.base_waypoints = None
        self.waypoint_tree = None
        self.waypoints_2d = None
        self.pose = None
        self.stopline_waypoint_index = -1

        # TODO: Add other member variables you need below

        self.loop()


    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # if the initialization is done
            if self.pose and self.base_waypoints and self.waypoint_tree:
                closest_waypoint_id = self.get_closest_waypoints()
                self.publish_waypoints( closest_waypoint_id )
            rate.sleep()

    
    def get_closest_waypoints(self):
        '''
        Returns the indices of the closest :py:data:LOOKAHEAD_WPS waypoints 
        to the last known pose :py:attr:pose
        '''
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_id = self.waypoint_tree.query( [x,y], 1 )[1]
        # coordinates of the waypoint behind the closest one 
        x1 = self.waypoints_2d[closest_id-1][0]
        y1 = self.waypoints_2d[closest_id-1][1]
        # coordinates of the closest waypoint
        x2 = self.waypoints_2d[closest_id][0]
        y2 = self.waypoints_2d[closest_id][1]
        #dot product of the vectors originating at the current pose
        #ending in the two waypoints above 
        dot_product = (x1-x) * (x2-x) + (y1 - y) * (y2 - y)
        if dot_product <= 0:
            #(x,y) is between (x1,y1) and (x2,y2)
            return closest_id
        else:
            # both are behind
            return (closest_id + 1) % len(self.waypoints_2d)
        
    def publish_waypoints(self, start):
        '''
        Publishes a message of type :py:class:Lane containing the points 
        ahead the last know pose :py:attr:pose.
        '''
        lane = Lane()
        
        closest_index = self.get_closest_waypoints()
        farthest_index = closest_index + LOOKAHEAD_WPS
        
        rospy.logdebug("closest_index = %d farthest_index =  %d self.stopline_waypoint_index = %d", 
                      closest_index, farthest_index, self.stopline_waypoint_index)
        
        lane.header = self.base_waypoints.header
        if self.stopline_waypoint_index == -1 or ( self.stopline_waypoint_index >= farthest_index ):
            lane.waypoints = self.base_waypoints.waypoints[ closest_index : farthest_index ]
        else:
            # decelerate
            lane.waypoints = []
            # stop a little before the stop line to account for falf of the length of the car
            stop_index = max( self.stopline_waypoint_index - 2, 0 )
            
            # try to stop smoothly in 30 waypoints from the stop line
            v_index = max(0, stop_index - 30)
            v0 = self.get_waypoint_velocity( self.base_waypoints.waypoints[v_index] )
            x0 = self.distance(self.base_waypoints.waypoints, v_index, stop_index)/2.0
            rospy.logdebug("x0=%f v0=%f", x0, v0)
            
            for i in range(closest_index, stop_index):
                waypoint = Waypoint()
                waypoint.pose = self.base_waypoints.waypoints[i].pose
                # calculate velocity as a sgmoid function
                dist = self.distance(self.base_waypoints.waypoints, i, stop_index)
                exp_term = math.exp( -0.2*(dist - x0) )
                velocity = v0 / (1+exp_term)
                if ( i == stop_index - 1 ):
                    velocity = 0.0
                rospy.logdebug("%f %f", dist, velocity)
                waypoint.twist.twist.linear.x = min( velocity, self.base_waypoints.waypoints[i].twist.twist.linear.x )
                lane.waypoints.append(waypoint)
            #set the velocity of the last point to 0.0
            #lane.waypoints[ len(lane.waypoints) - 1 ] = 0.0
                
        
        self.final_waypoints_pub.publish(lane)
        
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[ waypoint.pose.pose.position.x, waypoint.pose.pose.position.y ] 
                                 for waypoint in waypoints.waypoints ]
            self.waypoint_tree = KDTree(self.waypoints_2d) 

    def traffic_cb(self, msg):
        rospy.logdebug( "traffic_cb stopline_waypoint_index= %d", msg.data )
        self.stopline_waypoint_index = msg.data

    def obstacle_cb(self, msg):
        print( "obstacle_cb" )
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
