import io
import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image


class TritonPythonModel:
    def initialize(self, args):
        """Called once when the model is being loaded."""
        # Load the Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        from pruna import SmashConfig

        # Initialize the SmashConfig
        smash_config = SmashConfig()
        smash_config['compilers'] = ['step_caching']
        smash_config['comp_step_caching_interval'] = 3

        from pruna import smash

        # Smash the model
        self.smashed_model = smash(
            model=self.pipe,
            token="insert_your_token_here",  # TODO: insert your token here
            smash_config=smash_config,
        )
        
        # Parse model configuration
        self.model_config = json.loads(args["model_config"])
        
        # Get output data type
        output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Called for inference requests."""
        responses = []
        for request in requests:
            # Get input text
            input_text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_texts = input_text_tensor.as_numpy().astype(str).flatten().tolist()

            # Generate images
            generated_images = []
            for text in input_texts:
                # Generate image using the pipeline
                image = self.pipe(text).images[0]

                # Convert the PIL image to bytes (e.g., PNG format)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                generated_images.append(buffer.getvalue())

            # Convert the list of images to a numpy array of bytes
            output_array = np.array(generated_images, dtype=np.object_)

            # Create Triton output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """Called when the model is being unloaded."""
        print("Cleaning up...")
