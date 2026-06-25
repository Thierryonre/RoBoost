import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ColorFormat
from pathlib import Path
# from pprint import pprint
from ultralytics import YOLO

class Vision:
	def __init__(
		self,
		modelDir: str = "models",
		device: str = "intel:cpu",
		confidence: float = 0.35,
		iou: float = 0.70,
		imageSize: int = 640,
		onlyUseRelevantSegmentClasses: bool = False,
	):
		self.modelDir = Path(modelDir)
		self.modelDir.mkdir(parents=True, exist_ok=True)

		self.device = device
		self.confidence = confidence
		self.iou = iou
		self.imageSize = imageSize

		# Only objects with these labels are passed to the segmentation model.
		self.relevantSegmentClasses = ["bottle", "can"]
		self.canReplacementLabels = ["skateboard"]

		# Download and export the models when OpenVINO versions are unavailable.
		segmentationYOLOModelPath = self.modelDir / "yolo26m-seg.pt"
		segmentationVINOModelPath = self.modelDir / "yolo26m-seg_openvino_model/"

		if not segmentationVINOModelPath.exists():
			print("OpenVINO Segmentation model cannot be found - converting to OpenVINO now")
			model = YOLO(str(segmentationYOLOModelPath))
			model.export(format="openvino", half=True)

		# Load the OpenVINO Models
		print("Loading the OpenVINO models")
		self.segmentationModel = YOLO(str(segmentationVINOModelPath), task="segment")

	def createMasks(self, bgr):
		rawMasks = self.segmentationModel.predict(
			source=bgr,
			conf=self.confidence,
			iou=self.iou,
			imgsz=self.imageSize,
			device=self.device,
			retina_masks=True,
			verbose=False,
		)[0]

		masks = []
		for i in range(len(rawMasks.boxes)):
			label = rawMasks.names[int(rawMasks.boxes.cls[i].item())]
			box = rawMasks.boxes.xyxy[i].cpu().numpy() # [x1, y1, x2, y2]
			mask = rawMasks.masks.data[i].cpu().numpy()

			if label in self.canReplacementLabels:
				label = "can"

			if onlyUseRelevantSegmentClasses:
				if label not in self.relevantSegmentClasses:
					continue

			masks.append({
				"label": label,
				"box": box,
				"mask": mask
			})

		return masks, rawMasks

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

			centroidX, centroidY = np.round(centroidX).astype(int), np.round(centroidY).astype(int)

		return centroidX, centroidY

	def getAlignedFrames(self, camera):
		# Access the internal pipeline
		frameset = camera._pipeline.wait_for_frames()
		alignedFrameset = camera._align.process(frameset)

		# Get the raw SDK objects
		colorFrame = alignedFrameset.get_color_frame()
		depthFrame = alignedFrameset.get_depth_frame()

		return np.asanyarray(colorFrame.get_data()), depthFrame
