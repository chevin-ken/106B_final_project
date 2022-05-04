#!/usr/bin/env python

from baxter import * 
import rospy
import sys
import argparse

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

	datapath = "demo/"

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
			image = baxter.get_image()

            #TODO save as raw if training better with raw:
			save_path = datapath + "{}.jpeg".format(count)
			cv2.imwrite(light_save_path, light_image)

			camera_pose = baxter.get_left_camera_pose()
			camera_info_line = [count, camera_pose.translation.x,camera_pose.translation.y,camera_pose.translation.z,camera_pose.rotation.x,camera_pose.rotation.y,camera_pose.rotation.z,camera_pose.rotation.w]
			camera_lines.append(camera_info_line)
		except Exception as e:
			print(e)
		count += 1
	camera_file_path = datapath + "camera_positions.txt"
	with open(camera_file_path, "w+") as f:
		for line in camera_lines:
			f.write(str(line) + "\n")
	f.close()

def grasp(baxter, args, mesh):
    grasping_policy = GraspingPolicy(
        args.n_vert, args.n_grasps, args.n_execute, args.n_facets, args.metric
    )
    grasp_vertices_total, grasp_poses = grasping_policy.top_n_actions(mesh, args.obj)
    baxter.calibrate_gripper()
    rospy.sleep(2.0)
    for grasp_vertices, grasp_pose in zip(grasp_vertices_total, grasp_poses):
        grasping_policy.visualize_grasp(mesh, grasp_vertices, grasp_pose)
        if not args.sim:
            repeat = True
            while repeat:
                execute_grasp(grasp_pose, baxter)
                repeat = raw_input("repeat? [y|n] ") == 'y'

def execute_grasp(T_world_grasp, baxter):
    """
    Perform a pick and place procedure for the object. One strategy (which we have
    provided some starter code for) is to
    1. Move the gripper from its starting pose to some distance behind the object
    2. Move the gripper to the grasping pose
    3. Close the gripper
    4. Move up
    5. Place the object somewhere on the table
    6. Open the gripper. 
    As long as your procedure ends up picking up and placing the object somewhere
    else on the table, we consider this a success!
    HINT: We don't require anything fancy for path planning, so using the MoveIt
    API should suffice. Take a look at path_planner.py. The `plan_to_pose` and
    `execute_plan` functions should be useful. If you would like to be fancy,
    you can also explore the `compute_cartesian_path` functionality described in
    http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    
    Parameters
    ----------
    T_world_grasp : 4x4 :obj:`numpy.ndarray`
        pose of gripper relative to world frame when grasping object
    """
    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return

    # print(T_world_grasp)

    # T_world_grasp[:3, 3] += 0.007 * T_world_grasp[:3, 2]

    goal_x = T_world_grasp[0, 3]
    goal_y = T_world_grasp[1, 3] - 0.01
    goal_z = T_world_grasp[2, 3] - 0.009


    # print(T_world_grasp)
    # Go behind object
    # print("prepare to grip...")
    baxter.open_gripper()
    pose = Pose()
    xyz = T_world_grasp[:3, 3]
    xyz_offset = xyz - 0.05 * T_world_grasp[:3, 2]
    xyz_offset = xyz_offset.flatten()
    pose.position.x = xyz_offset[0]
    pose.position.y = xyz_offset[1]
    pose.position.z = xyz_offset[2]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_matrix(T_world_grasp[:3,:3])
    # plan = planner.plan_to_pose(pose)
    planner.change_velocity(0.5)
    plan = planner.plan_to_pose(pose)
    planner.execute_plan(plan)
    
    # Then swoop in
    print("go to grip position")
    pose.position.x = goal_x
    pose.position.y = goal_y
    pose.position.z = goal_z
    baxter.move_to_pose(pose, [], "right")

    raw_input()
    rospy.sleep(0.5)
    baxter.close_gripper()

    raw_input()

    # # Bring the object up
    pose.position.z += 0.1
    baxter.move_to_pose(pose, [], "right")

    raw_input()

    # # And over
    pose.position.y += 0.1
    baxter.move_to_pose(pose, [], "right")
    raw_input()

    # # And now place it
    pose.position.z -= 0.1
    baxter.move_to_pose(pose, [], "right")
    raw_input()
    baxter.open_gripper()

def parse_args():
    """
    Parses arguments from the user. Read comments for more details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_vert', type=int, default=1000, help=
        'How many vertices you want to sample on the object surface.  Default: 1000'
    )
    parser.add_argument('-n_facets', type=int, default=32, help=
        """You will approximate the friction cone as a set of n_facets vectors along 
        the surface.  This way, to check if a vector is within the friction cone, all 
        you have to do is check if that vector can be represented by a POSITIVE 
        linear combination of the n_facets vectors.  Default: 32"""
    )
    parser.add_argument('-n_grasps', type=int, default=500, help=
        'How many grasps you want to sample.  Default: 500')
    parser.add_argument('-n_execute', type=int, default=5, help=
        'How many grasps you want to execute.  Default: 5')
    parser.add_argument('-metric', '-m', type=str, default='compute_force_closure', help=
        """Which grasp metric in grasp_metrics.py to use.  
        Options: compute_force_closure, compute_gravity_resistance, compute_robust_force_closure"""
    )
    return parser.parse_args()

def load_mesh(filepath):
    mesh = trimesh.load_mesh(filepath)
    #Apply appropriate transformation

    #Load colmap camera pose
    colmap_camera_pose = None
    
    #Load corresponding baxter camera pose
    baxter_camera_pose = None

    #Calculate transformation based on poses
    transformation = None

    #Apply transformation to mesh
    mesh.apply_transform(transformation)
    mesh.fix_normals()
    return mesh
    
if __name__ == "__main__":
    args = parse_args()
	baxter = Baxter()
	baxter.change_velocity(0.5)
	collect_data(baxter)
    print("Dark images collected, press enter when Reconstruction stage is done")
    raw_input()

    #Load mesh from images
    mesh = load_mesh("demo/mesh.obj")
    grasp(baxter, args, mesh)
