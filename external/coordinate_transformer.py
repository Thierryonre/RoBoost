import yaml
import numpy as np
from robot_controller import (
    matrix_from_quaternion,
    make_homogeneous,
    matrix_from_euler
)

class CoordinateTransformer:
    """
    Handles math for coordinate frame transformations, specifically mapping 
    positions and orientations from the camera's frame to the robot base frame.
    """
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.T_ee_cam = None
        self.load_calibration()

    def load_calibration(self):
        """Loads hand-eye calibration parameters and constructs T_ee_cam."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            calib = config.get("hand_eye_calibration", {})
            translation = calib.get("translation", [0.0, 0.0, 0.0])
            q = calib.get("rotation_quaternion", [0.0, 0.0, 0.0, 1.0])
            
            # Construct rotation matrix from calibration quaternion
            R_ee_cam = matrix_from_quaternion(q)
            self.T_ee_cam = make_homogeneous(R_ee_cam, np.array(translation))
            
            print(f"[Transformer] Loaded hand-eye calibration:")
            print(f"  Translation: {translation}")
            print(f"  Rotation Q: {q}")
        except Exception as e:
            print(f"[Transformer] Error loading config/calibration: {e}. Defaulting to identity.")
            self.T_ee_cam = np.eye(4)

    def camera_to_base_pose(self, P_cam, theta_deg, T_base_ee):
        """
        Transforms a detected 3D point P_cam and 2D grasp angle theta_deg
        from the camera frame to the robot base frame, returning a 4x4 matrix.
        
        P_cam: 3D point in camera frame [X_c, Y_c, Z_c]
        theta_deg: 2D orientation angle in degrees (clockwise in image plane)
        T_base_ee: Current 4x4 end-effector pose matrix
        """
        # 1. Convert grasp angle (degrees) to rotation matrix in camera frame.
        # Since the camera is looking straight down, the rotation in the image 
        # is around the camera's Z axis.
        theta_rad = np.radians(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        
        # Rotation around camera Z-axis:
        R_c_grasp = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Homogeneous pose in camera frame
        T_c_grasp = make_homogeneous(R_c_grasp, np.array(P_cam))
        
        # 2. Map target pose to base frame:
        # T_base_target = T_base_ee * T_ee_cam * T_c_grasp
        T_base_target = T_base_ee @ self.T_ee_cam @ T_c_grasp
        
        # Apply standard gripper rotation offset if necessary.
        # Usually, when grasping from overhead, the gripper's Z-axis points 
        # downward (along the camera's positive Z-axis). Our R_c_grasp represents 
        # rotation around Z, which naturally aligns.
        
        return T_base_target
