import os
import re
from PIL import Image

try:
    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor
    import torch
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

class SmolVLAInference:
    """
    Handles Vision-Language-Action (VLA) inference using SmolVLA / SmolVLM
    optimized through OpenVINO. Provides text prompt formatting,
    model loading, execution, and output coordinate parsing.
    """
    def __init__(self, model_id_or_path="HuggingFaceTB/SmolVLM2-256M-Instruct", device="CPU", mock=False):
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.mock = mock or not OPENVINO_AVAILABLE
        
        self.processor = None
        self.model = None
        
        if self.mock:
            print("[VLA] Running in MOCK mode. OpenVINO VLA model will not be loaded.")
        else:
            self._load_model()

    def _load_model(self):
        """Loads the processor and OpenVINO visual causal LM model."""
        print(f"[VLA] Loading processor from '{self.model_id_or_path}'...")
        self.processor = AutoProcessor.from_pretrained(self.model_id_or_path)
        
        # Check if the path is already a directory with OpenVINO IR XML files
        is_ov_dir = False
        if os.path.isdir(self.model_id_or_path):
            files = os.listdir(self.model_id_or_path)
            if any(f.endswith(".xml") for f in files):
                is_ov_dir = True

        if is_ov_dir:
            print(f"[VLA] Loading pre-compiled OpenVINO model from directory '{self.model_id_or_path}' on device '{self.device}'...")
            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.model_id_or_path, 
                device=self.device
            )
        else:
            print(f"[VLA] Exporting & loading model '{self.model_id_or_path}' to OpenVINO on device '{self.device}'...")
            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.model_id_or_path, 
                export=True, 
                device=self.device
            )
        print("[VLA] OpenVINO model loaded successfully.")

    def run_inference(self, cv2_image, shape_name="triangle", object_type="peg", shape_options=None):
        """
        Executes SmolVLA inference on the provided image to locate the target shape.
        Returns the parsed coordinates (x, y, angle).
        """
        if shape_options is None:
            shape_options = ["circle", "triangle", "square"]
            
        # 1. Construct the prompt ensuring shape options and format constraints are explicitly included
        prompt = (
            f"Identify the center location and grasp angle of the {shape_name} {object_type}. "
            f"Choose the shape from options: {shape_options}. "
            f"Return the 2D center pixel coordinates [x, y] and rotation angle [angle] in degrees. "
            f"Provide your answer in the exact bracket format: [{object_type}: x, y, angle]"
        )
        
        print(f"[VLA] Shape Target: {shape_name} | Type: {object_type} | Options: {shape_options}")
        print(f"[VLA] Prompt: {prompt}")

        if self.mock:
            # Return realistic mock coordinates corresponding to the center of the triangle
            # peg in 'Triangle Peg.jpeg' and socket in 'Triangle Socket.jpeg' (assuming 1280x720)
            if object_type == "peg":
                mock_x, mock_y, mock_angle = 640, 360, 45.0  # Center peg, 45 degrees rotation
            else:
                mock_x, mock_y, mock_angle = 720, 400, 15.0  # Socket slightly off-center, 15 degrees
            
            print(f"[VLA] Mock output: [{object_type}: {mock_x}, {mock_y}, {mock_angle:.1f}]")
            return mock_x, mock_y, mock_angle

        # Convert CV2 BGR to PIL RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # 2. Build Chat Template Input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_input = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_input, images=pil_image, return_tensors="pt")

        # 3. Model generation
        output_tokens = self.model.generate(**inputs, max_new_tokens=60)
        decoded_output = self.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        print(f"[VLA] Model raw output: {decoded_output}")
        
        # 4. Parse output
        parsed = self.parse_coordinates(decoded_output, object_type)
        if parsed is not None:
            return parsed
        else:
            raise ValueError(f"[VLA] Failed to parse coordinates from model output: '{decoded_output}'")

    def parse_coordinates(self, text, object_type):
        """
        Parses text output to extract coordinates [x, y, angle].
        Looks for variations of bracketed coordinates.
        """
        # Search for pattern like [peg: x, y, angle] or [x, y, angle]
        # Match integers or floats
        pattern = r"\[\s*(?:" + re.escape(object_type) + r"\s*:\s*)?(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if not match:
            # Fallback pattern for any general [x, y, angle] format
            fallback_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
            match = re.search(fallback_pattern, text)
            
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                angle = float(match.group(3))
                return x, y, angle
            except ValueError:
                pass
                
        return None
