#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Face Mask Cropping Tool - Crops masks based on face detection in corresponding images
'''
import argparse
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Crop face masks based on face detection in corresponding images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing masks (.npy or .png)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped masks')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for face detection')
    parser.add_argument('--facecrop_dilation_ratio', type=float, default=0.5,
                        help='Dilation ratio for face crop area')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Output mask size (width and height)')
    parser.add_argument('--mask_format', type=str, choices=['npy', 'png', 'both'], default='both',
                        help='Format of mask files to process')
    return parser.parse_args()

def select_face(det_bboxes, probs):
    """Select the main face from detected faces"""
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

def crop_and_pad(img, rect):
    """Crop an image based on a rectangle, ensuring a square crop"""
    x0, y0, x1, y1 = rect
    h, w = img.shape[:2]

    # Ensure coordinates are within image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    # Calculate original width and height
    width = x1 - x0
    height = y1 - y0

    # Use the smaller dimension as the side length for the square
    side_length = min(width, height)

    # Calculate center point of the rectangle
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # Calculate new square coordinates centered on the original rectangle
    new_x0 = max(0, center_x - side_length // 2)
    new_y0 = max(0, center_y - side_length // 2)
    new_x1 = min(w, new_x0 + side_length)
    new_y1 = min(h, new_y0 + side_length)

    # Ensure final crop is square
    if (new_x1 - new_x0) != (new_y1 - new_y0):
        side_length = min(new_x1 - new_x0, new_y1 - new_y0)
        new_x1 = new_x0 + side_length
        new_y1 = new_y0 + side_length

    # Crop the image
    cropped_image = img[new_y0:new_y1, new_x0:new_x1]

    return cropped_image, (new_x0, new_y0, new_x1, new_y1)

def process_mask(image_path, mask_dir, output_dir, face_detector, args):
    """Process a single image and its corresponding mask"""
    # try:
    # Load image for face detection
    face_img = cv2.imread(image_path)
    if face_img is None:
        print(f"Could not read image: {image_path}")
        return False
        
    # Create paths for corresponding mask files
    image_name = Path(image_path).stem
    mask_npy_path = os.path.join(mask_dir, f"{image_name}.npy")
    mask_png_path = os.path.join(mask_dir, f"{image_name}.png")
    
    mask_data = None
    mask_format = None
    
    # Try to load mask based on specified format
    if args.mask_format in ['npy', 'both'] and os.path.exists(mask_npy_path):
        mask_data = np.load(mask_npy_path)
        mask_format = 'npy'
        # Print debug info
        print(f"Loaded NPY mask: shape={mask_data.shape}, dtype={mask_data.dtype}")
    elif args.mask_format in ['png', 'both'] and os.path.exists(mask_png_path):
        mask_data = cv2.imread(mask_png_path)
        mask_format = 'png'
    
    if mask_data is None:
        print(f"No corresponding mask found for {image_name}")
        return False
    
    # Ensure mask is in a format OpenCV can handle
    if mask_format == 'npy':
        # Convert to appropriate format for OpenCV
        if len(mask_data.shape) == 2:  # If 2D array
            # Convert to 3-channel if needed
            mask_data = np.repeat(mask_data[:, :, np.newaxis], 3, axis=2)
        
        # Ensure correct data type
        if mask_data.dtype != np.uint8:
            # Normalize to 0-255 range for display
            if mask_data.max() <= 1.0:
                mask_data = (mask_data * 255).astype(np.uint8)
            else:
                mask_data = mask_data.astype(np.uint8)
        
    # Detect face in the image
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    
    if select_bbox is None:
        print(f"No face detected in {image_path}")
        return False
    
    # Extract face coordinates and apply dilation
    xyxy = select_bbox[:4]
    xyxy = np.round(xyxy).astype('int')
    rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]

    # Save face bounding box coordinates to a .npy file
    face_coords = np.array([rb, re, cb, ce])
    coords_output_dir = os.path.join(os.path.dirname(output_dir), "face_coords")
    os.makedirs(coords_output_dir, exist_ok=True)
    coords_output_path = os.path.join(coords_output_dir, f"{image_name}.npy")
    np.save(coords_output_path, face_coords)
    print(f"Saved face coordinates to {coords_output_path}")
    
    # Face crop with dilation
    r_pad_crop = int((re - rb) * args.facecrop_dilation_ratio)
    c_pad_crop = int((ce - cb) * args.facecrop_dilation_ratio)
    crop_rect = [
        max(0, cb - c_pad_crop),
        max(0, rb - r_pad_crop),
        min(ce + c_pad_crop, face_img.shape[1]),
        min(re + r_pad_crop, face_img.shape[0])
    ]
    
    # Verify crop dimensions
    print(f"Crop rectangle: {crop_rect}")
    if crop_rect[2] <= crop_rect[0] or crop_rect[3] <= crop_rect[1]:
        print(f"Invalid crop rectangle for {image_name}")
        return False
    
    # Crop the mask using the same crop rectangle
    cropped_mask, _ = crop_and_pad(mask_data, crop_rect)
    
    # Check cropped mask
    if cropped_mask.size == 0:
        print(f"Cropped mask is empty for {image_name}")
        return False
        
    print(f"Cropped mask: shape={cropped_mask.shape}, dtype={cropped_mask.dtype}")
    
    # Resize the cropped mask to the output size
    resized_mask = cv2.resize(cropped_mask, (args.output_size, args.output_size))
    
    # Save the cropped mask
    output_path = os.path.join(output_dir, f"{image_name}")
    if mask_format == 'npy':
        # If it was originally npy, save as npy (with original shape)
        if len(resized_mask.shape) == 3 and resized_mask.shape[2] == 3:
            # Convert back to original format if needed
            resized_mask = resized_mask[:, :, 0]  # Take first channel if all the same
        np.save(f"{output_path}.npy", resized_mask)
    else:
        cv2.imwrite(f"{output_path}.png", resized_mask)
        
    return True
        
    # except Exception as e:
    #     print(f"Error processing {image_path}: {str(e)}")
    #     return False

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize face detector
    face_detector = MTCNN(
        image_size=320,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=args.device
    )
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Looking for corresponding masks in {args.mask_dir}")
    
    # Process each image and its corresponding mask
    successful = 0
    for image_path in tqdm(image_files, desc="Processing masks"):
        if process_mask(image_path, args.mask_dir, output_dir, face_detector, args):
            successful += 1
        else:
            print(f"Failed to process mask for {Path(image_path).stem}")
    
    print(f"Processing complete. Successfully cropped {successful}/{len(image_files)} masks.")
    print(f"Cropped masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()