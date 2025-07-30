# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model, 
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg 
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
import cv2
import numpy as np

import sys
# Step 1: Get the parent directory path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Step 2: Append the parent directory to sys.path
sys.path.append(parent_directory)

from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video


class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    
    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb


def batch_inference_process(args: argparse.Namespace):
    """
    Perform batch inference processing on multiple (image, audio) pairs.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    
    This function processes multiple audio files with their corresponding images
    based on the RAVDESS naming convention.
    """
    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    
    # Get the directory containing audio files
    audio_dir = config.audio_dir
    image_dir = config.image_dir
    save_path = config.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get all audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Process each audio file
    for audio_file in audio_files:
        print(f"Processing {audio_file}")
        
        # Parse the filename to extract actor ID
        # Format: XX-XX-XX-XX-XX-XX-XX.wav
        parts = audio_file.split('-')
        if len(parts) != 7:
            print(f"Skipping {audio_file}: Invalid filename format")
            continue
        
        actor_id = parts[6].split('.')[0]  # Get the actor ID (last part before extension)
        
        # Determine the corresponding image file
        image_file = f"subject_{actor_id}.jpg"
        image_path = os.path.join(image_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"Skipping {audio_file}: Image {image_file} not found")
            continue
        
        # Set up paths for this pair
        audio_path = os.path.join(audio_dir, audio_file)
        output_filename = f"output_{audio_file.replace('.wav', '.mp4')}"
        output_path = os.path.join(save_path, output_filename)
        
        # Create a copy of the config for this specific pair
        pair_config = OmegaConf.create(OmegaConf.to_container(config))
        pair_config.source_image = image_path
        pair_config.driving_audio = audio_path
        pair_config.output = output_path
        
        # Create a new args namespace with the updated config
        pair_args = argparse.Namespace(**OmegaConf.to_container(pair_config))
        
        # Run inference for this pair
        try:
            output_file = inference_process(pair_args)
            print(f"Successfully processed {audio_file} with {image_file}. Output saved to {output_file}")
        except Exception as e:
            print(f"Error processing {audio_file} with {image_file}: {str(e)}")
    
    print("Batch processing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_image", type=str, required=False,
                        help="source image")
    parser.add_argument("--driving_audio", type=str, required=False,
                        help="driving audio")
    parser.add_argument(
        "--output", type=str, help="output video file name", default=".cache/output.mp4")
    parser.add_argument(
        "--pose_weight", type=float, help="weight of pose", required=False)
    parser.add_argument(
        "--face_weight", type=float, help="weight of face", required=False)
    parser.add_argument(
        "--lip_weight", type=float, help="weight of lip", required=False)
    parser.add_argument(
        "--face_expand_ratio", type=float, help="face region", required=False)
    parser.add_argument(
        "--audio_ckpt_dir", "--checkpoint", type=str, help="specific checkpoint dir", required=False)
    
    # Add new arguments for batch processing
    parser.add_argument(
        "--batch", action="store_true", help="enable batch processing mode")
    parser.add_argument(
        "--audio_dir", type=str, help="directory containing audio files for batch processing")
    parser.add_argument(
        "--image_dir", type=str, help="directory containing image files for batch processing")
    parser.add_argument(
        "--save_path", type=str, help="directory to save output videos", default=".cache/outputs")

    command_line_args = parser.parse_args()

    if command_line_args.batch:
        batch_inference_process(command_line_args)
    else:
        inference_process(command_line_args)
