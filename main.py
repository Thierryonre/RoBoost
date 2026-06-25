import cv2
import numpy as np
import pyrealsense2 as rs
from libraries.arm import Arm
from libraries.vision import Vision
from physicalai.capture import RealSenseCamera
from physicalai.capture.camera import ColorMode
from pprint import pprint
from threading import Thread

LOGGING = False
LIFT_DISTANCE = 0.1

def main():
	# Declare variables
	itemsInGlobalView = []

	# Load everything
	if LOGGING: print("Loading")
	arm = Arm()
	vision = Vision(onlyUseRelevantSegmentClasses=False)

	if LOGGING: print("Initialising camera")
	with RealSenseCamera(
		# serial_number="130322270884",	# Loose camera
		serial_number="130322272460",	# Mounted camera
		width=640,
		height=480,
		fps=15,			# [90/60/30/15/5] rs-enumerate-devices
		color_mode=ColorMode.BGR
	) as camera:
		while True:
			# Start the thread for finding all of the objects
			# Thread(target=, daemon=True).start()
			itemsInGlobalView = [{"coords": 0}]

			for item in itemsInGlobalView:
				# Approach the item
				if LOGGING: print("Approaching item")
				arm.approach(item["coords"])

				# Get a frame from the camera
				if LOGGING: print("Getting camera rgbd data")
				bgr, depth = vision.getAlignedFrames(camera)

				# Find the mask of the object
				if LOGGING: print("Creating the mask")
				masks, rawMasks = vision.createMasks(bgr)
				annotatedResult = rawMasks.plot()
				cv2.imwrite("images/outputWithMask.jpg", annotatedResult)

				# Display the image (optional)
				if LOGGING: print("Showing the image")
				cv2.imwrite("cameraTest.png", bgr)

				# Add the centroid of each mask and its depth to the masks
				for i, mask in enumerate(masks):
					masks[i]["centroid"] = centroid = centroidX, centroidY = vision.getCentroid(mask)
					masks[i]["centroidDepth"] = centroidDepth = depth.get_distance(centroidX, centroidY)
				continue

				# Create a grasp for the arm and lift it upwards
				pose = arm.createGrasp(mask, centroidDepth)
				arm.executePose(pose)
				liftedPose = arm.translatePoseY(pose, LIFT_DISTANCE)
				arm.executePose(liftedPose)

				# Find the weight of the item
				weight = arm.calculateWeight()

				# Classify and place the arm at the correct location depending on its weight
				if 0 <= weight < 200:		# light
					arm.executeWeightPose("light")
				elif 200 <= weight < 600:	# medium
					arm.executeWeightPose("medium")
				elif 600 <= weight:			# heavy
					arm.executeWeightPose("heavy")
				else:
					raise Exception(f"The weight was calculated to be {weight}g - this is not a valid weight")

				# Release
				arm.executePose(pose)
				arm.release()

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
