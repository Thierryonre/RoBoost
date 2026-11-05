import os
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class RealSenseCamera:
    """
    Wrapper for Intel RealSense D405 Depth Camera.
    Handles streaming, depth-color alignment, intrinsic calibration,
    fisheye/distortion correction, and 2D-to-3D back-projection.
    """
    def __init__(self, width=1280, height=720, fps=30, device_id="", mock=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.device_id = device_id
        self.mock = mock or not REALSENSE_AVAILABLE
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.intrinsics = None
        
        # Camera matrices for manual distortion correction if needed
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        if self.mock:
            print("[Camera] Running in MOCK mode. Physical RealSense camera will not be accessed.")
            self._init_mock()
        else:
            self._init_hardware()

    def _init_hardware(self):
        """Initializes the physical RealSense D405 camera pipeline."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if self.device_id:
            self.config.enable_device(self.device_id)
            
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # Start streaming
        profile = self.pipeline.start(self.config)
        
        # Create an align object to align depth stream to color stream
        # This aligns the geometry of color and depth frames and corrects for lens distortion
        self.align = rs.align(rs.stream.color)
        
        # Retrieve camera intrinsics for the aligned color stream
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        
        # Construct OpenCV camera matrix and distortion coefficients
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.distortion_coeffs = np.array(self.intrinsics.coeffs, dtype=np.float32)
        
        print(f"[Camera] Intel RealSense D405 initialized successfully.")
        print(f"[Camera] Intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, ppx={self.intrinsics.ppx:.2f}, ppy={self.intrinsics.ppy:.2f}")

    def _init_mock(self):
        """Initializes mock properties for testing without hardware."""
        # Standard placeholder intrinsics for a typical D405 camera
        self.intrinsics = type('Intrinsics', (object,), {
            'width': self.width,
            'height': self.height,
            'fx': 600.0,
            'fy': 600.0,
            'ppx': self.width / 2.0,
            'ppy': self.height / 2.0,
            'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
        })()
        
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.distortion_coeffs = np.zeros(5, dtype=np.float32)
        
        # Load local images for simulation
        self.mock_peg_img = None
        self.mock_socket_img = None
        
        if os.path.exists("Triangle Peg.jpeg"):
            self.mock_peg_img = cv2.imread("Triangle Peg.jpeg")
            self.mock_peg_img = cv2.resize(self.mock_peg_img, (self.width, self.height))
        if os.path.exists("Triangle Socket.jpeg"):
            self.mock_socket_img = cv2.imread("Triangle Socket.jpeg")
            self.mock_socket_img = cv2.resize(self.mock_socket_img, (self.width, self.height))

    def get_rgbd_frame(self, target_type="peg"):
        """
        Captures a frame and returns the aligned color, depth images.
        For hardware, it pulls a fresh aligned frame.
        For mock, it returns the local peg or socket image and a simulated depth map.
        
        target_type: "peg" or "socket" (used in mock mode to return the corresponding image)
        """
        if self.mock:
            # Simulate a depth map (e.g. constant 0.4 meters distance for objects on a table)
            depth_image = np.ones((self.height, self.width), dtype=np.float32) * 0.4
            
            if target_type == "peg" and self.mock_peg_img is not None:
                color_image = self.mock_peg_img.copy()
            elif target_type == "socket" and self.mock_socket_img is not None:
                color_image = self.mock_socket_img.copy()
            else:
                # Default black frame if files not found
                color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
            return color_image, depth_image, None

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("[Camera] Failed to acquire aligned depth/color frames.")
            
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Convert depth to meters (D405 depth unit is typically 0.001 or 0.0001 meters)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_in_meters = depth_image.astype(np.float32) * depth_scale
        
        return color_image, depth_in_meters, depth_frame

    def undistort_image(self, color_image):
        """
        Manually undistorts the color image using retrieved camera intrinsics.
        Note: Aligned RealSense frames are already rectified, but this provides 
        an explicit distortion correction filter if required.
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return color_image
        return cv2.undistort(color_image, self.camera_matrix, self.distortion_coeffs)

    def deproject_pixel_to_point(self, x, y, depth_value):
        """
        Converts 2D pixel coordinates (x, y) and its depth value (z)
        to a 3D point [X, Y, Z] in the camera coordinate frame.
        """
        if self.mock:
            # Simple analytical deprojection for mock mode
            X = (x - self.intrinsics.ppx) * depth_value / self.intrinsics.fx
            Y = (y - self.intrinsics.ppy) * depth_value / self.intrinsics.fy
            Z = depth_value
            return [X, Y, Z]
            
        # Physical deprojection using pyrealsense2
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_value)
        return point_3d

    def stop(self):
        """Stops the camera stream."""
        if not self.mock and self.pipeline is not None:
            self.pipeline.stop()
            print("[Camera] RealSense pipeline stopped.")
