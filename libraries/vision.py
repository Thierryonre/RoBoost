import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class Vision:
	def __init__(
		self,
		modelDir: str = "models",
		device: str = "intel:cpu",
		confidence: float = 0.25,
		iou: float = 0.70,
		imageSize: int = 1280,
		onlyUseRelevantSegmentClasses: bool = False,
	):
		self.modelDir = Path(modelDir)
		self.modelDir.mkdir(parents=True, exist_ok=True)

		self.device = device
		self.confidence = confidence
		self.iou = iou
		self.imageSize = imageSize
		self.onlyUseRelevantSegmentClasses = onlyUseRelevantSegmentClasses

		# Only objects with these labels are passed to the segmentation model
		self.relevantSegmentClasses = ["bottle", "can"]
		self.canReplacementLabels = ["skateboard"]
		self.segmentClassesToIgnore = ["person", "suitcase", "traffic light"]

		# Download and export the models when OpenVINO versions are unavailable
		segmentationYOLOModelPath = self.modelDir / "yolo26l-seg.pt"
		segmentationVINOModelPath = self.modelDir / "yolo26l-seg_openvino_model/"

		if not segmentationVINOModelPath.exists():
			print("OpenVINO Segmentation model cannot be found - converting to OpenVINO now")
			model = YOLO(str(segmentationYOLOModelPath))
			model.export(format="openvino", imgsz=self.imageSize, half=True)

		# Load the OpenVINO Models
		print("Loading the OpenVINO models")
		self.segmentationModel = YOLO(str(segmentationVINOModelPath), task="segment")

	def createMasks(self, bgr):
		# rawMasks = self.segmentationModel.predict(
		raw_results = self.segmentationModel.predict(
			source=bgr,
			conf=self.confidence,
			iou=self.iou,
			imgsz=self.imageSize,
			device=self.device,
			retina_masks=True,
			verbose=False,
		)[0]

		# 2. Determine which indices to keep and build mapping
		keep_indices = []
		# Create a local copy of names to modify for the plot
		new_names = raw_results.names.copy()

		for i in range(len(raw_results.boxes)):
			cls_id = int(raw_results.boxes.cls[i].item())
			label = raw_results.names[cls_id]

			# Apply your logic
			if label in self.canReplacementLabels:
				label = "can"

			# Apply filtering rules
			if self.onlyUseRelevantSegmentClasses and label not in self.relevantSegmentClasses:
				continue
			if label in self.segmentClassesToIgnore:
				continue

			# Update the name mapping for this class ID to your custom label
			new_names[cls_id] = label
			keep_indices.append(i)

		# 3. Filter the Results object in-place
		raw_results.boxes = raw_results.boxes[keep_indices]
		if raw_results.masks is not None:
			raw_results.masks.data = raw_results.masks.data[keep_indices]

		# Update the names dictionary so .plot() uses your labels
		raw_results.names = new_names

		# 4. Prepare your custom masks list (same as before)
		masks = []
		for i in keep_indices:
			masks.append({
				"label": new_names[int(raw_results.boxes.cls[0].item())], # uses the updated name
				"box": raw_results.boxes.xyxy[0].cpu().numpy(),
				"mask": raw_results.masks.data[0].cpu().numpy()
			})

		return masks, raw_results

		# masks = []
		# for i in range(len(rawMasks.boxes)):
		# 	label = rawMasks.names[int(rawMasks.boxes.cls[i].item())]
		# 	box = rawMasks.boxes.xyxy[i].cpu().numpy() # [x1, y1, x2, y2]
		# 	mask = rawMasks.masks.data[i].cpu().numpy()

		# 	if label in self.canReplacementLabels:
		# 		label = "can"

		# 	if self.onlyUseRelevantSegmentClasses:
		# 		if label not in self.relevantSegmentClasses:
		# 			continue

		# 	if label in self.segmentClassesToIgnore:
		# 		continue

		# 	masks.append({
		# 		"label": label,
		# 		"box": box,
		# 		"mask": mask
		# 	})

		# return masks, rawMasks

	def getCentroid(self, mask):
		# Crop the image to the ROI
		x1, y1, x2, y2 = np.round(mask["box"]).astype(int)
		croppedMask = mask["mask"][y1:y2, x1:x2]

		# Find the centroid within the cropped box
		coords = np.argwhere(croppedMask > 0.5)
		if coords.size > 0:
			# These coordinates are local to the crop (0,0 is the top-left of the box)
			croppedMaskCentroidY, croppedMaskCentroidX = coords.mean(axis=0)

			# Convert back to global coordinates for your analysis
			centroidX = x1 + croppedMaskCentroidX
			centroidY = y1 + croppedMaskCentroidY

			centroidX, centroidY = int(np.round(centroidX)), int(np.round(centroidY))

		return centroidX, centroidY

	def getAlignedFrames(self, camera):
		# Access the internal pipeline
		frameset = camera._pipeline.wait_for_frames()
		alignedFrameset = camera._align.process(frameset)

		# Get the raw SDK objects
		colorFrame = alignedFrameset.get_color_frame()
		depthFrame = alignedFrameset.get_depth_frame()

		return np.asanyarray(colorFrame.get_data()), depthFrame

	def removeInvalidDepthMasks(self, masks, rawMasks):
		keepIndices = [
			i for i, mask in enumerate(masks)
			if mask.get("centroidDepth") != 0.0
		]


		filteredMasks = [masks[i] for i in keepIndices]
		# print(keepIndices, filteredMasks)


		# rawMasks.boxes = rawMasks.boxes[keepIndices]

		# if rawMasks.masks is not None:
		# 	rawMasks.masks.data = rawMasks.masks.data[keepIndices]

		return masks, rawMasks
