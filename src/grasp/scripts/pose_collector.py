#!/usr/bin/env python

try:
	import rospy
	from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, Quaternion
	import tf2_ros
except:
	print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
	ros_enabled = False

def lookup_transform(target_frame, source_frame):
        tfBuffer = tf2_ros.Buffer()
        tfListener = tf2_ros.TransformListener(tfBuffer)
        r = rospy.Rate(10) # 10hz
        done = False
        trans = None
        while not rospy.is_shutdown() and not done:
            try:
                trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
                done = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print("Exception: {}".format(e))
            r.sleep()
        return trans

def collect_poses():
	num_poses = 5
	for i in range(num_poses):
		print("Move to next pose")
		raw_input()
		pose = lookup_transform("base", "left_gripper").transform
		print(pose)

if __name__ == "__main__":
	rospy.init_node('pose_collector')
	collect_poses()
