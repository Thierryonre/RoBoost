import argparse
import sys
import yaml
import numpy as np
from camera import RealSenseCamera
from vla_inference import SmolVLAInference
from robot_controller import (
    PandaRobotController,
    matrix_from_euler,
    make_homogeneous
)
from coordinate_transformer import CoordinateTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Franka Panda Peg-in-Socket Placement Pipeline")
    parser.add_argument(
        "--shape", 
        type=str, 
        default="triangle", 
        choices=["circle", "triangle", "square"],
        help="Shape of the peg and socket to match."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default=None, 
        choices=["ros2", "mujoco", "mock"],
        help="Execution mode (overrides config.yaml if provided)."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to the config.yaml file."
    )
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline(args):
    # 1. Load configuration
    print(f"[Main] Loading configuration from '{args.config}'...")
    config = load_config(args.config)
    
    # Overwrite mode if specified via CLI
    mode = args.mode if args.mode else config["robot"]["mode"]
    shape = args.shape
    shape_options = config["vla"]["shape_options"]
    
    print("=" * 60)
    print(f" PANDA SHAPE PLACEMENT PIPELINE (Shape: {shape.upper()} | Mode: {mode.upper()})")
    print("=" * 60)

    # 2. Initialize subsystems
    # Initialize Camera
    cam_cfg = config["camera"]
    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
        device_id=cam_cfg["device_id"],
        mock=(mode == "mock")
    )
    
    # Initialize Coordinate Transformer
    transformer = CoordinateTransformer(config_path=args.config)
    
    # Initialize Robot Controller
    robot_cfg = config["robot"]
    robot = PandaRobotController(
        mode=mode,
        base_frame=robot_cfg["base_frame"],
        ee_frame=robot_cfg["ee_frame"],
        robot_name=robot_cfg.get("robot_name", "panda")
    )
    
    # Initialize SmolVLA Inference
    vla_cfg = config["vla"]
    vla = SmolVLAInference(
        model_id_or_path=vla_cfg["model_id_or_path"],
        device=vla_cfg["device"],
        mock=(mode == "mock")
    )

    try:
        # 3. Execution Pipeline
        
        # Step 3.1: Open the robot's gripper
        robot.open_gripper()
        
        # Step 3.2: Move to overhead scanning pose
        print(f"\n[Step 1] Moving to overhead scanning pose...")
        scan_pose = robot_cfg["scan_pose"]
        R_scan = matrix_from_euler(scan_pose[3], scan_pose[4], scan_pose[5])
        t_scan = np.array(scan_pose[:3])
        T_scan = make_homogeneous(R_scan, t_scan)
        robot.move_to_pose(T_scan)
        
        # Step 3.3: Capture image & run SmolVLA to detect the peg
        print(f"\n[Step 2] Scanning workspace for the {shape} peg...")
        color_img, depth_img, depth_frame = camera.get_rgbd_frame(target_type="peg")
        
        # Optional manual lens distortion correction
        corrected_color = camera.undistort_image(color_img)
        
        # Run SmolVLA to get 2D pixel coordinates and orientation
        px_x, px_y, angle = vla.run_inference(
            corrected_color, 
            shape_name=shape, 
            object_type="peg", 
            shape_options=shape_options
        )
        
        # Get depth value at coordinates
        depth_val = depth_img[px_y, px_x]
        print(f"[Main] Peg detected at pixel [{px_x}, {px_y}] with depth {depth_val:.3f}m and orientation {angle:.1f} degrees.")
        
        if depth_val <= 0.0:
            print("[Warning] Depth sensor returned invalid reading. Using fallback table depth of 0.4m.")
            depth_val = 0.4
            
        # Project 2D pixel to 3D point in camera frame
        P_cam = camera.deproject_pixel_to_point(px_x, px_y, depth_val)
        print(f"[Main] Projected 3D point in camera frame: {P_cam}")
        
        # Get current end-effector pose to transform coordinate to robot base frame
        T_base_ee = robot.get_current_ee_pose()
        T_base_grasp = transformer.camera_to_base_pose(P_cam, angle, T_base_ee)
        print(f"[Main] Calculated grasp target pose in base frame:\nPosition: {T_base_grasp[:3, 3]}\nRotation Matrix:\n{T_base_grasp[:3, :3]}")

        # Step 3.4: Execute Pick
        print(f"\n[Step 3] Executing pick sequence for {shape} peg...")
        
        # Define pre-grasp pose (offset along robot's base Z frame)
        T_pre_grasp = T_base_grasp.copy()
        T_pre_grasp[2, 3] += robot_cfg["pre_grasp_offset_z"]
        
        # Move to pre-grasp pose
        print(f"[Main] Moving to pre-grasp pose (Z-offset: +{robot_cfg['pre_grasp_offset_z']}m)...")
        robot.move_to_pose(T_pre_grasp)
        
        # Open gripper just in case
        robot.open_gripper()
        
        # Descend to actual grasp pose
        T_grasp = T_base_grasp.copy()
        # Add a tiny adjustment parameter if needed to not crash into table
        T_grasp[2, 3] += robot_cfg["grasp_depth_z"]
        
        print(f"[Main] Descending to grasp pose (Z-height: {T_grasp[2, 3]:.3f}m)...")
        robot.move_to_pose(T_grasp)
        
        # Close gripper to grasp peg
        # D405 peg width is roughly 3cm (0.03m)
        robot.close_gripper(grasp_width=0.03, force=15.0)
        
        # Lift peg back to pre-grasp pose
        print("[Main] Lifting peg...")
        robot.move_to_pose(T_pre_grasp)

        # Step 3.5: Return to scan pose to locate socket
        print(f"\n[Step 4] Returning to overhead scan pose...")
        robot.move_to_pose(T_scan)

        # Step 3.6: Scan workspace for socket
        print(f"\n[Step 5] Scanning workspace for the {shape} socket...")
        color_img, depth_img, depth_frame = camera.get_rgbd_frame(target_type="socket")
        corrected_color = camera.undistort_image(color_img)
        
        # Run SmolVLA to detect socket location and orientation
        px_x, px_y, angle = vla.run_inference(
            corrected_color, 
            shape_name=shape, 
            object_type="socket", 
            shape_options=shape_options
        )
        
        # Retrieve depth
        depth_val = depth_img[px_y, px_x]
        print(f"[Main] Socket detected at pixel [{px_x}, {px_y}] with depth {depth_val:.3f}m and orientation {angle:.1f} degrees.")
        
        if depth_val <= 0.0:
            print("[Warning] Depth sensor returned invalid reading. Using fallback table depth of 0.4m.")
            depth_val = 0.4
            
        # Project socket to 3D point in camera frame
        P_cam = camera.deproject_pixel_to_point(px_x, px_y, depth_val)
        print(f"[Main] Projected socket 3D point in camera frame: {P_cam}")
        
        # Compute place/insertion target pose in base frame
        T_base_ee = robot.get_current_ee_pose()
        T_base_insert = transformer.camera_to_base_pose(P_cam, angle, T_base_ee)
        print(f"[Main] Calculated socket insertion target pose in base frame:\nPosition: {T_base_insert[:3, 3]}\nRotation Matrix:\n{T_base_insert[:3, :3]}")

        # Step 3.7: Execute Place / Insertion
        print(f"\n[Step 6] Executing insertion sequence for {shape} peg...")
        
        # Define pre-insertion pose (offset along base Z frame)
        T_pre_insert = T_base_insert.copy()
        T_pre_insert[2, 3] += robot_cfg["insert_offset_z"]
        
        # Move to pre-insertion pose
        print(f"[Main] Moving to pre-insertion pose (Z-offset: +{robot_cfg['insert_offset_z']}m)...")
        robot.move_to_pose(T_pre_insert)
        
        # Descend into socket
        T_insert = T_base_insert.copy()
        T_insert[2, 3] += robot_cfg["insert_depth_z"]
        
        print(f"[Main] Descending into socket (Z-height: {T_insert[2, 3]:.3f}m)...")
        robot.move_to_pose(T_insert)
        
        # Open gripper to release the peg
        robot.open_gripper()
        
        # Retreat back to pre-insertion pose
        print("[Main] Retreating gripper...")
        robot.move_to_pose(T_pre_insert)

        # Move to scan pose to finish clean
        print("\n[Step 7] Returning to default scanning pose...")
        robot.move_to_pose(T_scan)
        
        print("\n" + "=" * 60)
        print(" PIPELINE EXECUTED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n[Main] Shutting down camera and robot interfaces...")
        camera.stop()
        robot.shutdown()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
