import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

# Connect to TritonServer
client = InferenceServerClient(url="localhost:8001")

# Input text for the model
input_text = np.array(["a turtle", "a futuristic city"], dtype=object).reshape(-1, 1)

# Create the input tensor
input_tensor = InferInput("INPUT_TEXT", input_text.shape, "BYTES")
input_tensor.set_data_from_numpy(input_text)

# Specify the output tensor
output = InferRequestedOutput("OUTPUT")

# Perform inference with only the text input
response = client.infer(
    model_name="stable_diffusion",
    inputs=[input_tensor],
    outputs=[output]
)

# Retrieve and process the output
output_data = response.as_numpy("OUTPUT")
print(f"Generated image shapes: {output_data.shape}")
