import numpy as np
import onnx
import onnxruntime as ort
from typing import List, Dict
from deploy_robot.policies.policy import Policy

class OnnxPolicy(Policy):
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the OnnxPolicy.

        Args:
            model_path (str): Path to the ONNX model file.
            device (str): Device to run the ONNX model on. Options are "cpu" or "cuda".
        """
        super().__init__()
        self.model_path = model_path

        # Configure ONNX Runtime session options based on the device
        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.output_names = [out.name for out in self.ort_session.get_outputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.ort_session.get_inputs()}
        self.output_shapes = {out.name: out.shape for out in self.ort_session.get_outputs()}

        # Initialize input and output tensors
        self.input_tensors: Dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.input_shapes.items()
        }
        self.output_tensors: Dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.output_shapes.items()
        }

        # Load metadata and update properties
        self._load_metadata()

    def get_observation_size(self) -> int:
        shape = self.input_shapes.get("obs")
        return shape[1] if shape else 0

    def get_action_size(self) -> int:
        shape = self.output_shapes.get("actions")
        return shape[1] if shape else 0

    def reset(self):
        action_size = self.get_action_size()
        self.output_tensors["actions"] = np.zeros((1, action_size))

    def forward(self, observations: np.ndarray, output_names: List[str] = ["actions"]) -> np.ndarray:
        """
        Perform a forward pass through the ONNX model.

        Args:
            observations (np.ndarray): The observation input corresponding to "obs".
                - If 1D, it will be reshaped to 2D by adding a batch dimension.
            output_names (List[str]): A list of output names to compute. Defaults to ["actions"].

        Returns:
            np.ndarray: The last action computed by the model (scaled by action scale) if "actions" is in outputs.
        """
        # Ensure "obs" is provided
        if "obs" not in self.input_names:
            raise ValueError("The model does not have an input named 'obs'.")

        # Ensure observations is 2D (batch size, features)
        if observations.ndim == 1:
            observations = np.expand_dims(observations, axis=0)  # Add batch dimension

        if observations.ndim != 2:
            raise ValueError(f"'obs' input must be 2D (batch_size, features). Got shape: {observations.shape}")

        # Prepare inputs for the ONNX model
        inputs = {"obs": observations}
        for name in self.input_names:
            if name == "obs":
                continue
            # Use existing input tensor or default to zeros
            inputs[name] = self.input_tensors[name]

        # Run partial inference
        self.run_partial_inference(inputs, output_names)

        # Return the scaled action if "actions" is in outputs
        if "actions" in output_names:
            return self.get_last_action()

    def get_unscaled_last_action(self) -> np.ndarray:
        """
        Get the last computed action without scaling.

        Returns:
            np.ndarray: The unscaled action.
        """
        actions = self.output_tensors.get("actions", np.zeros((1, self.get_action_size())))
        return actions[0]  # Return the first (and usually only) batch element

    def get_last_action(self) -> np.ndarray:
        """
        Get the last computed action, scaled by the action scale.

        Returns:
            np.ndarray: The scaled action.
        """
        actions = self.output_tensors.get("actions", np.zeros((1, self.get_action_size())))
        return actions[0] * self.action_scale

    def run_partial_inference(self, inputs: Dict[str, np.ndarray], output_names: List[str]) -> None:
        """
        Run inference on the ONNX model with specified inputs and outputs.

        Args:
            inputs (Dict[str, np.ndarray]): A dictionary mapping input names to their corresponding numpy arrays.
            output_names (List[str]): A list of output names to compute.

        Returns:
            None: Updates the specified outputs in `self.output_tensors`.
        """
        # Validate inputs
        for name in inputs.keys():
            if name not in self.input_names:
                raise ValueError(f"Input '{name}' is not a valid input for the model. Valid inputs: {self.input_names}")

        # Validate outputs
        for name in output_names:
            if name not in self.output_names:
                raise ValueError(f"Output '{name}' is not a valid output for the model. Valid outputs: {self.output_names}")

        # Run inference
        results = self.ort_session.run(output_names, inputs)

        # Update only the specified outputs in `self.output_tensors`
        for name, result in zip(output_names, results):
            self.output_tensors[name] = result

    def _load_metadata(self):
        """Private method to load metadata and update properties."""
        model = onnx.load(self.model_path)
        metadata = {prop.key: prop.value for prop in model.metadata_props}

        # Update properties from metadata
        self.joint_names = metadata.get("joint_names", "").split(",")
        self.joint_stiffness = np.array([float(x) for x in metadata.get("joint_stiffness", "").split(",")])
        self.joint_damping = np.array([float(x) for x in metadata.get("joint_damping", "").split(",")])
        self.default_joint_positions = np.array([float(x) for x in metadata.get("default_joint_pos", "").split(",")])
        self.command_names = metadata.get("command_names", "").split(",")
        self.observation_names = metadata.get("observation_names", "").split(",")
        self.action_scale = np.array([float(x) for x in metadata.get("action_scale", "").split(",")])

if __name__ == "__main__":
    # Replace this with the path to your ONNX model
    model_path = "model/dalafan_prog.onnx"

    # Initialize the OnnxPolicy with CPU
    print("Initializing policy on CPU...")
    policy_cpu = OnnxPolicy(model_path, device="cpu")

    # Initialize the OnnxPolicy with GPU (if available)
    print("Initializing policy on GPU...")
    policy_gpu = OnnxPolicy(model_path, device="cuda")

    # Print input and output information for the CPU policy
    print("\nModel Inputs (CPU):")
    for name, shape in policy_cpu.input_shapes.items():
        print(f"  {name}: shape={shape}")

    print("\nModel Outputs (CPU):")
    for name, shape in policy_cpu.output_shapes.items():
        print(f"  {name}: shape={shape}")

    print("\nCommand Names:")
    print(policy_cpu.command_names)
    print("\nObservation Names:")
    print(policy_cpu.observation_names)

    # Perform a simple inference test
    print("\nRunning inference on CPU...")
    if "obs" in policy_cpu.input_names:
        obs_shape = policy_cpu.input_shapes["obs"]
        dummy_obs = np.random.rand(*obs_shape).astype(np.float32)
        actions = policy_cpu.forward(dummy_obs)
        print(f"Actions (CPU): {actions}")
    else:
        print("No 'obs' input found in the model.")