from threading import Thread
from libraries.arm import Arm
from libraries.vision import Vision

def main():
	# Declare variables
	itemsInGlobalView = []

	# Load everything
	arm = Arm()
	vision = Vision()

	with RealSenseCamera(
		serial_number=serialNumber,
		width=640,
		height=480,
		fps=10,
	) as camera:
		while True:
			# Start the thread for finding all of the objects
			Thread(target=, daemon=True).start()

			for item in itemsInGlobalView:
				# Approach the item
				arm.approach["coords"]
				
				# Get a frame from the camera
	  		  	rgb, depth = camera.read_rgbd()

				# Find the mask of the object
				mask = vision.createMask(rgb)

				# Find the centroid of the mask
				centroid = vision.getCentroid(mask)

				# Get the depth of the centroid
				centroidDepth = vision.findDepth(centroid)

				# Create a grasp for the arm and lift it upwards
				pose = arm.createGrasp(mask, centroidDepth)
				arm.executePose(pose)
				liftedPose = arm.translatePoseY(pose, distance)
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
	main()
