#!/usr/bin/env python

from baxter import * 
import rospy
import sys

from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from moveit_msgs.msg import Constraints, OrientationConstraint
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
ros_enabled = True

def get_r_from_quaternion(quaternion):
	quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
	r = R.from_quat(quaternion)
	return r

def create_transform(origin, direction):
	"""
	Creates a homogenous transformation matrix at the origin such that the 
	z axis is the same as the direction specified. There are infinitely 
	many of such matrices, but we choose the one where the y axis is as 
	vertical as possible.  
	Parameters
	----------
	origin (3x np.ndarray): origin coordinates
	direction (3x np.ndarray): direction vector
	Returns
	-------
	(4x4 np.ndarray): homogenous transform matrix
	"""
	up = np.array([0, 0, 1])
	y = normalize(direction) # create a z vector in the given direction
	x = normalize(np.cross(up, y)) # create a x vector perpendicular to z and up
	z = -np.cross(y, x) # create a y vector perpendicular to z and x

	# print('\n\n===== NEW GRASP =====\n')
	# print('initial generation:')
	# print(x, y, z)
	# raw_input()

	result = np.eye(4)

	# set rotation part of matrix
	result[0:3,0] = x
	result[0:3,1] = y
	result[0:3,2] = z

	# set translation part of matrix to origin
	# result[0:3,3] = origin

	return result

def get_quaternion_from_r(r):
	r_quat = r.as_quat()
	quaternion = Quaternion()
	quaternion.x = r_quat[0]
	quaternion.y = r_quat[1]
	quaternion.z = r_quat[2]
	quaternion.w = r_quat[3]

	return quaternion

def normalize(v):
	return v/np.linalg.norm(v)

def rot_z(theta, rot_mat):
	Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
	return np.matmul(Rz, rot_mat)

def collect_data(baxter):
	num_pics = 50
	z_table = -0.25

	datapath = "data/"
	object_name = "allegra/"

	#move to initial_postion 
	print("Move gripper to initial position")
	raw_input()
	starting_pose = baxter.get_left_hand_pose()
	z_circle = starting_pose.translation.z
	z_difference = z_circle - z_table
	starting_pos = np.array([starting_pose.translation.x, starting_pose.translation.y, z_circle])

	starting_orientation = starting_pose.rotation
	starting_rotation_matrix = get_r_from_quaternion(starting_orientation).as_dcm()

	y = starting_rotation_matrix[0:3, 2]

	#determine object position
	obj_pos = starting_pos + (y/y[2]) * -z_difference
	r = np.sqrt(np.sum((starting_pos[:2] - obj_pos[:2]) ** 2))

	count = 0
	camera_lines = []
	for t in np.linspace(0, 2 * np.pi, num_pics):
		try:
			print("Image pair number {}".format(count))
			theta = -t
			new_rot = rot_z(theta, starting_rotation_matrix)
			new_orientation = get_quaternion_from_r(R.from_dcm(new_rot))

			new_direction = new_rot[0:3, 2]
			new_position = obj_pos + (new_direction/new_direction[2]) * z_difference

			new_pose = Pose()
			new_pose.position.x = new_position[0]
			new_pose.position.y = new_position[1]
			new_pose.position.z = z_circle
			new_pose.orientation = new_orientation
			# new_pose.orientation.w = 1

			baxter.move_to_pose(new_pose, [], "left")

			print("Press enter to take light image")
			raw_input()
			light_image = baxter.get_image()

			print("Press enter to take dark image")
			raw_input()
			dark_image = baxter.get_image()

			light_save_path = datapath + object_name + "light/{}.jpeg".format(count)
			cv2.imwrite(light_save_path, light_image)
			dark_save_path = datapath + object_name + "dark/{}.jpeg".format(count)
			cv2.imwrite(dark_save_path, dark_image)

			camera_pose = baxter.get_left_camera_pose()
			camera_info_line = [count, camera_pose.translation.x,camera_pose.translation.y,camera_pose.translation.z,camera_pose.rotation.x,camera_pose.rotation.y,camera_pose.rotation.z,camera_pose.rotation.w]
			camera_lines.append(camera_info_line)
		except Exception as e:
			print(e)
		count += 1
	camera_file_path = datapath + object_name + "camera_positions.txt"
	with open(camera_file_path, "w+") as f:
		for line in camera_lines:
			f.write(str(line) + "\n")
	f.close()
if __name__ == "__main__":
	baxter = Baxter()
	baxter.change_velocity(0.5)
	collect_data(baxter)
