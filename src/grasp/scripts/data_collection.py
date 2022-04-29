#!/usr/bin/env python


try:
	import rospy
	import cv2
	from cv_bridge import CvBridge
	from sensor_msgs.msg import Image
	ros_enabled = True

except:
	print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
	ros_enabled = False

bridge = CvBridge()
camera_image_topic = '/cameras/left_hand_camera/image'


def take_screenshots():
	try:
		print("Press enter to take screenshot for light image")
		raw_input()
		image = rospy.wait_for_message(camera_image_topic, Image)
		print("Converting image to CV2")
		light_cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

		print("Press enter to take screenshot for dark image")
		raw_input()
		image = rospy.wait_for_message(camera_image_topic, Image)
		print("Converting image to CV2")
		dark_cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

		return light_cv_image, dark_cv_image
	except Exception as e:
		print(e)

def take_images():
	datapath = "data/"
	object_name = "cube/"
	start_index = 0
	end_index = 1
	for i in range(start_index, end_index):
		print("Image number {}".format(start_index))
		light_image, dark_image = take_screenshots()
		print("Saving images")
		light_save_path = datapath + object_name + "light/{}.jpeg".format(i)
		cv2.imwrite(light_save_path, light_image)
		dark_save_path = datapath + object_name + "dark/{}.jpeg".format(i)
		cv2.imwrite(dark_save_path, dark_image)

if __name__ == "__main__":
	rospy.init_node('dummy_tf_node')
	take_images()