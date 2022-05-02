from planner import PathPlanner
import rospy, cv2, cv_bridge
from sensor_msgs.msg import Image
from baxter_interface.camera import CameraController
from baxter_interface import Limb
from baxter_interface import gripper as robot_gripper
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform, Quaternion
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.transform import Rotation as R
import numpy as np
from moveit_msgs.msg import Constraints, OrientationConstraint

def open_cam(camera, res):
    # Check if valid resolution
    if not any((res[0] == r[0] and res[1] == r[1])for r in CameraController.MODES):
        rospy.logerr("Invalid resolution provided.")
        # Open camera
        cam = CameraController(camera)
        # Create camera object
        cam.resolution = res
        # Set resolution
        cam.open()  
        # open# Close camera

def close_cam(camera):
    cam = CameraController(camera)  # Create camera object
    cam.close() # close

def get_r_from_quaternion(quaternion):
    quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    r = R.from_quat(quaternion)
    return r

def get_quaternion_from_r(r):
    r_quat = r.as_quat()
    quaternion = Quaternion()
    quaternion.x = r_quat[0]
    quaternion.y = r_quat[1]
    quaternion.z = r_quat[2]
    quaternion.w = r_quat[3]

    return quaternion

# takes in Transform object (not TransformStamped), returns 4x4 homogenous transform matrix
def make_homog_from_transform(transform): 
    quaternion = transform.rotation
    translation = transform.translation
    # rpy =  euler_from_quaternion(quaternion) #given in native TF API
    quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    translation = [translation.x, translation.y, translation.z]
    r = R.from_quat(quaternion)
    rot_mat = r.as_dcm()

    homog = np.zeros((4, 4))
    homog[:3, :3] = np.array(rot_mat)
    homog[3] = np.array([0, 0, 0, 1])
    homog[:3, 3] = np.array(translation).T
    return homog

def get_pose_from_homog(g_ba, g_al):
    g_bl = np.dot(g_ba, g_al)
    pose = Pose()
    r = R.from_dcm(g_bl[:3, :3])
    quat = r.as_quat()
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    pose.position.x = g_bl[0][3]
    pose.position.y = g_bl[1][3]
    pose.position.z = g_bl[2][3]
    return pose

class Baxter:
    def camera_callback(self, image):
        self.image = image

    def setup_left_hand_camera(self):
        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.pose.position.x = 0.737
        pose.pose.position.y = 0.219
        pose.pose.position.z = 0.107
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 1
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 0
        orientation_constraints = []
        plan = self.plan(pose, orientation_constraints, "left")
        raw_input("Press <Enter> to move the camera to initial pose: ")
        self.execute(plan, "left")

    def setup_table_obstacle(self):
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base"
        table_pose.pose.position.x = 0.5
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.3
        table_pose.pose.orientation.x = 0.0
        table_pose.pose.orientation.y = 0.0
        table_pose.pose.orientation.z = 0.0
        table_pose.pose.orientation.w = 1.0

        self.left_planner.add_box_obstacle(np.array([0.40, 1.20, 0.10]), "table", table_pose)

    def remove_table_obstacle(self):
        self.left_planner.remove_obstacle("table")

    def calibrate_gripper(self):
        self.left_gripper.calibrate()

    def __init__(self):
        self.image = None

        self.left_planner = PathPlanner("left_arm")

        rospy.init_node('Baxter')
        # self.remove_table_obstacle()
        self.setup_table_obstacle()

        self.scan_again = True
        rospy.Subscriber('cameras/left_hand_camera/image', Image, self.camera_callback)

        while(self.image == None):
            continue

        self.left_gripper = robot_gripper.Gripper('left')
        self.left_gripper.open()

        self.left_gripper.calibrate()
        rospy.sleep(2.0)

        self.bridge = cv_bridge.CvBridge()
        print("Finished init")

    def get_image(self):
        return self.bridge.imgmsg_to_cv2(self.image, desired_encoding="passthrough")

    def get_left_hand_pose(self):
        return self.lookup_transform("base", "left_gripper").transform

    def get_left_camera_pose(self):
        return self.lookup_transform("base", "left_hand_camera").transform

    def close_gripper(self):
        self.left_gripper.close()

    def open_gripper(self):
        self.left_gripper.open()

    def rescan(self):
        raw_input("Press enter to scan again")
        self.scan_again = True

    def change_velocity(self, scaling_factor):
        self.left_planner.change_velocity(scaling_factor)

    def plan(self, target, orientation_constraints, arm):
        if arm == "left":
            return self.left_planner.plan_to_pose(target, orientation_constraints)
        else:
            return self.right_planner.plan_to_pose(target, orientation_constraints)

    def execute(self, plan, arm):
        if arm == "left":
            return self.left_planner.execute_plan(plan)
        else:
            return self.right_planner.execute_plan(plan)

    def move_to_pose(self, pose, orientation_constraints, gripper):
        success = False
        while not success:
            plan_found = False
            while not plan_found:
                try:
                    plan = self.plan(pose, orientation_constraints, gripper)
                    plan_found = len(plan.joint_trajectory.points) > 0
                except:
                    print("Plan not found, continuing to search")
                if not plan_found:
                    print("Plan not found, continuing to search")
            raw_input("Press enter to move to pose")
            success = self.execute(plan, gripper)

    def lookup_transform(self, target_frame, source_frame):
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

        return trans # of type TransformStamped

    def move_forward(self, forward_amount):
        target_trans = self.get_left_hand_pose()
        target_pose = Pose()
        target_pose.orientation = target_trans.rotation
        target_pose.position = target_trans.translation
        current_orientation = target_pose.orientation
        target_pose.position.x = target_pose.position.x + forward_amount
        orientation_constraint = OrientationConstraint()
        orientation_constraint.link_name = "left_gripper"
        orientation_constraint.header.frame_id = "base"
        orientation_constraint.orientation = current_orientation
        orientation_constraint.absolute_x_axis_tolerance = .05
        orientation_constraint.absolute_y_axis_tolerance = .05
        orientation_constraint.absolute_z_axis_tolerance = .05

        return self.plan(target_pose, [orientation_constraint], "left")

    def move_up(self, up_amount):
        target_trans = self.get_left_hand_pose()
        target_pose = Pose()
        target_pose.orientation = target_trans.rotation
        target_pose.position = target_trans.translation
        current_orientation = target_pose.orientation
        target_pose.position.z = target_pose.position.z + up_amount
        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = "base"
        target_pose_stamped.pose = target_pose
        return self.plan(target_pose_stamped, [], "left")
            
    def test(self):
        return



if __name__ == '__main__':
    b = Baxter()
    b.test()