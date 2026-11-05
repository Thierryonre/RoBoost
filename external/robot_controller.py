import os
import time
import numpy as np

# Try importing ROS 2 libraries
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from geometry_msgs.msg import PoseStamped as PoseStamped2
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Try importing MuJoCo
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Try importing ROS 2 Gripper Actions
GRASP_ACTION_TYPE = None
MOVE_ACTION_TYPE = None
if ROS2_AVAILABLE:
    try:
        from franka_msgs.action import Grasp as FrankaGraspAction
        from franka_msgs.action import Move as FrankaMoveAction
        GRASP_ACTION_TYPE = FrankaGraspAction
        MOVE_ACTION_TYPE = FrankaMoveAction
    except ImportError:
        try:
            from control_msgs.action import GripperCommand as ControlGripperCommand
            GRASP_ACTION_TYPE = ControlGripperCommand
        except ImportError:
            pass


# ==========================================
# Standalone Geometry & Transformation Helpers
# ==========================================

def quaternion_from_matrix(R):
    """Converts a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

def matrix_from_quaternion(q):
    """Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def matrix_from_euler(roll, pitch, yaw):
    """Converts Euler angles (roll, pitch, yaw) in radians to a 3x3 rotation matrix."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def make_homogeneous(R, t):
    """Creates a 4x4 homogeneous transformation matrix from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ==========================================
# Robot Controller Implementation
# ==========================================

class PandaRobotController:
    """
    Standardized driver interface for Franka Emika Panda.
    Supports ROS 2, MuJoCo direct simulation, and Mock mode.
    IncludesNamespaced topics & actions compatible with multipanda_ros2.
    """
    def __init__(self, mode="mock", base_frame="panda_link0", ee_frame="panda_hand", robot_name="panda"):
        self.mode = mode.lower()
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.robot_name = robot_name
        
        self.current_pose = None
        self.gripper_open = True
        
        # Initialize internal state for Mock mode (defaults to scanning pose)
        if self.mode == "mock":
            print("[Robot] Initializing in MOCK mode.")
            R = matrix_from_euler(np.pi, 0.0, 0.0)
            t = np.array([0.4, 0.0, 0.5])
            self.current_pose = make_homogeneous(R, t)
        elif self.mode == "mujoco":
            self._init_mujoco()
        elif self.mode == "ros2":
            if not ROS2_AVAILABLE:
                raise ImportError("[Robot] rclpy is not installed. Cannot use ros2 mode.")
            self._init_ros2()
        else:
            raise ValueError(f"[Robot] Unknown mode: {self.mode}. Select from: [ros2, mujoco, mock]")

    def _init_mujoco(self):
        """Initializes direct MuJoCo simulation environment connector."""
        print("[Robot] Initializing MuJoCo simulation interface...")
        self.mj_model = None
        self.mj_data = None
        
        # Start current pose tracking (defaults to scan pose)
        R = matrix_from_euler(np.pi, 0.0, 0.0)
        t = np.array([0.4, 0.0, 0.5])
        self.current_pose = make_homogeneous(R, t)
        
        if MUJOCO_AVAILABLE:
            xml_path = "panda_mujoco.xml"
            if os.path.exists(xml_path):
                try:
                    self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
                    self.mj_data = mujoco.MjData(self.mj_model)
                    print(f"[Robot] Loaded MuJoCo model successfully from '{xml_path}'.")
                except Exception as e:
                    print(f"[Robot] Failed to load MuJoCo XML: {e}")
            else:
                print(f"[Robot] Model file '{xml_path}' not found. Initialized in virtual headless mock simulator.")
        else:
            print("[Robot] 'mujoco' python library not installed. Running in virtual fallback mode.")

    def _init_ros2(self):
        """Initializes ROS 2 publishers, listeners, and action clients for multipanda_ros2."""
        print(f"[Robot] Initializing ROS 2 interface for namespace '/{self.robot_name}'...")
        rclpy.init(args=None)
        self.node = Node(f"{self.robot_name}_peg_placement_controller")
        
        # ROS 2 publisher using multipanda_ros2 naming scheme
        pose_topic = f"/{self.robot_name}/cartesian_pose_controller/commands"
        self.pose_pub = self.node.create_publisher(PoseStamped2, pose_topic, 10)
        print(f"[Robot] Created pose command publisher on '{pose_topic}'.")
        
        # ROS 2 TF listener
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        
        # ROS 2 Action Clients for Gripper using namespaced actions
        self.grasp_client = None
        self.move_client = None
        
        if GRASP_ACTION_TYPE is not None:
            grasp_action_path = f"/{self.robot_name}/panda_hand_controller/grasp"
            self.grasp_client = ActionClient(self.node, GRASP_ACTION_TYPE, grasp_action_path)
            print(f"[Robot] Gripper grasp action client initialized on '{grasp_action_path}'.")
            
        if MOVE_ACTION_TYPE is not None:
            move_action_path = f"/{self.robot_name}/panda_hand_controller/move"
            self.move_client = ActionClient(self.node, MOVE_ACTION_TYPE, move_action_path)
            print(f"[Robot] Gripper move action client initialized on '{move_action_path}'.")
            
        print("[Robot] ROS 2 interface fully initialized.")

    def get_current_ee_pose(self):
        """Retrieves the current end-effector pose T_base_ee (4x4 matrix)."""
        if self.mode == "mock":
            return self.current_pose.copy()
            
        elif self.mode == "mujoco":
            if self.mj_model is not None and self.mj_data is not None:
                try:
                    # Find body ID of end effector (e.g. ee_frame)
                    body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ee_frame)
                    if body_id != -1:
                        pos = self.mj_data.xpos[body_id]
                        mat = self.mj_data.xmat[body_id].reshape(3, 3)
                        self.current_pose = make_homogeneous(mat, pos)
                except Exception as e:
                    print(f"[Robot] MuJoCo TF Error: {e}")
            return self.current_pose.copy()
                
        elif self.mode == "ros2":
            try:
                now = rclpy.time.Time()
                trans = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, now, rclpy.duration.Duration(seconds=1.0))
                translation = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
                rotation = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
                R = matrix_from_quaternion(rotation)
                return make_homogeneous(R, np.array(translation))
            except Exception as e:
                print(f"[Robot] ROS 2 TF Error: {e}")
                return np.eye(4)

    def move_to_pose(self, target_pose_matrix):
        """Moves the robot end-effector to the target 4x4 pose matrix."""
        translation = target_pose_matrix[:3, 3]
        rotation_matrix = target_pose_matrix[:3, :3]
        q = quaternion_from_matrix(rotation_matrix)
        
        print(f"[Robot] Commanding movement to Position: {translation}, Quaternion: {q}")
        
        if self.mode in ["mock", "mujoco"]:
            # Simulate physical movement and update local track state
            time.sleep(1.0)
            self.current_pose = target_pose_matrix.copy()
            
            # Step the simulation in MuJoCo if model is running
            if self.mode == "mujoco" and self.mj_model is not None and self.mj_data is not None:
                try:
                    # Drive joints to targets (simplified as direct setting)
                    # Real simulators would use inverse kinematics or joints PD controllers
                    mujoco.mj_step(self.mj_model, self.mj_data)
                except Exception as e:
                    print(f"[Robot] MuJoCo step error: {e}")
            print(f"[Robot] Motion complete ({self.mode} mode).")
            
        elif self.mode == "ros2":
            msg = PoseStamped2()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            msg.pose.position.x = translation[0]
            msg.pose.position.y = translation[1]
            msg.pose.position.z = translation[2]
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
            self.pose_pub.publish(msg)
            time.sleep(1.5)

    def _send_gripper_action_goal(self, client, goal_msg):
        """Safely sends action goal to the namespaced ROS 2 action client."""
        if client is None:
            print("[Robot] Action client is not available. Skipping command.")
            return False
            
        print(f"[Robot] Connecting to action server '{client._action_name}'...")
        if not client.wait_for_server(timeout_sec=2.0):
            print(f"[Robot] Timed out waiting for action server '{client._action_name}'.")
            return False
            
        send_goal_future = client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=2.0)
        goal_handle = send_goal_future.result()
        
        if goal_handle is None or not goal_handle.accepted:
            print("[Robot] Action goal rejected.")
            return False
            
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, get_result_future, timeout_sec=2.0)
        return True

    def open_gripper(self):
        """Opens the gripper."""
        print("[Robot] Opening gripper...")
        if self.mode in ["mock", "mujoco"]:
            if self.mode == "mujoco" and self.mj_model is not None and self.mj_data is not None:
                # MuJoCo gripper control: set finger actuators to maximum width (0.04m)
                for i in range(self.mj_model.nu):
                    name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if "finger" in name or "gripper" in name:
                        self.mj_data.ctrl[i] = 0.04
                mujoco.mj_step(self.mj_model, self.mj_data)
            time.sleep(0.5)
            self.gripper_open = True
            print("[Robot] Gripper opened.")
            
        elif self.mode == "ros2":
            if self.move_client is not None:
                goal_msg = MOVE_ACTION_TYPE.Goal()
                goal_msg.width = 0.08  # fully open
                goal_msg.speed = 0.1
                success = self._send_gripper_action_goal(self.move_client, goal_msg)
                if success:
                    self.gripper_open = True
                    print("[Robot] ROS 2 Gripper opened.")
            else:
                # Fallback to standard control commands if move action not present
                print("[Robot] Warning: Gripper MoveAction client not available.")
                self.gripper_open = True

    def close_gripper(self, grasp_width=0.03, speed=0.05, force=10.0):
        """Closes the gripper to grasp an object."""
        print(f"[Robot] Closing gripper (Grasp width: {grasp_width}m, Force: {force}N)...")
        if self.mode in ["mock", "mujoco"]:
            if self.mode == "mujoco" and self.mj_model is not None and self.mj_data is not None:
                # Set finger actuators to grasp position (half of target width per finger)
                finger_pos = grasp_width / 2.0
                for i in range(self.mj_model.nu):
                    name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if "finger" in name or "gripper" in name:
                        self.mj_data.ctrl[i] = finger_pos
                mujoco.mj_step(self.mj_model, self.mj_data)
            time.sleep(0.5)
            self.gripper_open = False
            print("[Robot] Gripper closed.")
            
        elif self.mode == "ros2":
            if self.grasp_client is not None:
                goal_msg = GRASP_ACTION_TYPE.Goal()
                
                # Check message fields to support both franka_msgs and control_msgs
                if hasattr(goal_msg, 'width'):
                    # franka_msgs Grasp
                    goal_msg.width = grasp_width
                    goal_msg.epsilon.inner = 0.005
                    goal_msg.epsilon.outer = 0.005
                    goal_msg.speed = speed
                    goal_msg.force = force
                elif hasattr(goal_msg, 'command'):
                    # control_msgs GripperCommand
                    goal_msg.command.position = grasp_width
                    goal_msg.command.max_effort = force
                    
                success = self._send_gripper_action_goal(self.grasp_client, goal_msg)
                if success:
                    self.gripper_open = False
                    print("[Robot] ROS 2 Gripper closed/grasped.")
            else:
                print("[Robot] Warning: Gripper GraspAction client not available.")
                self.gripper_open = False

    def shutdown(self):
        """Cleanup."""
        if self.mode == "ros2":
            rclpy.shutdown()
