#!/usr/bin/env python3
"""
Data preprocessing script for the Food Recognition System
This script processes raw food image datasets and prepares them for training.
"""

import os
import argparse
import shutil
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess food image datasets')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (food-101, uec-food, etc.)')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directory containing raw dataset')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--num_classes', type=int, default=0, help='Limit to top N classes (0 for all)')
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples per class (0 for all)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    return parser.parse_args()

def process_food101(data_dir, output_dir, train_ratio=0.8, img_size=224, num_classes=0, max_samples=0, workers=4):
    """Process the Food-101 dataset"""
    print(f"Processing Food-101 dataset from {data_dir}...")
    
    # Define paths
    food101_dir = os.path.join(data_dir, 'food-101')
    images_dir = os.path.join(food101_dir, 'images')
    meta_dir = os.path.join(food101_dir, 'meta')
    
    # Check if dataset exists
    if not os.path.exists(food101_dir):
        raise FileNotFoundError(f"Food-101 dataset not found in {food101_dir}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load class list
    with open(os.path.join(meta_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Limit classes if specified
    if num_classes > 0:
        classes = classes[:num_classes]
    
    print(f"Processing {len(classes)} food classes")
    
    # Load train/test splits
    with open(os.path.join(meta_dir, 'train.json'), 'r') as f:
        train_files = json.load(f)
    
    with open(os.path.join(meta_dir, 'test.json'), 'r') as f:
        test_files = json.load(f)
    
    # Combine and shuffle for custom split
    all_files = {}
    for cls in classes:
        class_files = train_files.get(cls, []) + test_files.get(cls, [])
        random.shuffle(class_files)
        
        # Limit samples if specified
        if max_samples > 0:
            class_files = class_files[:max_samples]
        
        # Calculate split
        split_idx = int(len(class_files) * train_ratio)
        all_files[cls] = {
            'train': class_files[:split_idx],
            'val': class_files[split_idx:]
        }
    
    # Process images
    annotations = []
    
    def process_image(args):
        cls, img_file, split = args
        src_path = os.path.join(images_dir, f"{img_file}.jpg")
        
        if not os.path.exists(src_path):
            return None
        
        # Create class directory if it doesn't exist
        dst_dir = os.path.join(train_dir if split == 'train' else val_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Construct destination path
        dst_path = os.path.join(dst_dir, f"{img_file.split('/')[-1]}.jpg")
        
        try:
            # Open, resize, and convert image
            img = Image.open(src_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.LANCZOS)
            
            # Save processed image
            img.save(dst_path, 'JPEG', quality=95)
            
            # Return annotation
            return {
                'filename': os.path.relpath(dst_path, output_dir),
                'class': cls,
                'split': split
            }
        except Exception as e:
            print(f"Error processing {src_path}: {str(e)}")
            return None
    
    # Prepare arguments for parallel processing
    process_args = []
    for cls in classes:
        for split in ['train', 'val']:
            for img_file in all_files[cls][split]:
                process_args.append((cls, img_file, split))
    
    # Process images in parallel
    print(f"Processing {len(process_args)} images with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_image, process_args), total=len(process_args)))
    
    # Filter out failed processing
    annotations = [r for r in results if r is not None]
    
    # Save annotations to CSV
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(os.path.join(output_dir, 'annotations.csv'), index=False)
    
    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Print statistics
    train_count = len(annotations_df[annotations_df['split'] == 'train'])
    val_count = len(annotations_df[annotations_df['split'] == 'val'])
    print(f"Processed {train_count + val_count} images")
    print(f"Training set: {train_count} images")
    print(f"Validation set: {val_count} images")

def process_uecfood(data_dir, output_dir, train_ratio=0.8, img_size=224, num_classes=0, max_samples=0, workers=4):
    """Process the UEC Food dataset"""
    print(f"Processing UEC Food dataset from {data_dir}...")
    
    # Define paths
    uecfood_dir = os.path.join(data_dir, 'UECFood')
    
    # Check if dataset exists
    if not os.path.exists(uecfood_dir):
        raise FileNotFoundError(f"UEC Food dataset not found in {uecfood_dir}")
    
    # Implementation for UEC Food dataset
    # Similar structure to food101 processing but adapted for UEC Food
    print("UEC Food dataset processing not fully implemented")

def create_sample_images(output_dir, num_samples=5):
    """Create sample images for demo purposes"""
    print("Creating sample images for demo...")
    
    # Create samples directory
    samples_dir = os.path.join(output_dir, '../samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get random images from processed data
    train_dir = os.path.join(output_dir, 'train')
    classes = os.listdir(train_dir)
    
    for cls in random.sample(classes, min(num_samples, len(classes))):
        class_dir = os.path.join(train_dir, cls)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            if images:
                # Copy a random image to samples
                src_image = os.path.join(class_dir, random.choice(images))
                dst_image = os.path.join(samples_dir, f"{cls}.jpg")
                shutil.copy(src_image, dst_image)
                print(f"Created sample for {cls}")

def create_visualization(output_dir):
    """Create visualization of the dataset"""
    print("Creating dataset visualization...")
    
    # Load annotations
    annotations_df = pd.read_csv(os.path.join(output_dir, 'annotations.csv'))
    
    # Count samples per class
    class_counts = annotations_df.groupby(['class', 'split']).size().unstack(fill_value=0)
    
    # Plot distribution
    plt.figure(figsize=(12, 8))
    class_counts.plot(kind='bar', stacked=True)
    plt.title('Number of images per class in the dataset')
    plt.xlabel('Food Class')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    
    # Create a grid of sample images
    classes = annotations_df['class'].unique()
    num_classes = min(5, len(classes))
    
    plt.figure(figsize=(15, 3 * num_classes))
    for i, cls in enumerate(classes[:num_classes]):
        # Get 5 random images from this class
        class_samples = annotations_df[annotations_df['class'] == cls]['filename'].values
        samples = np.random.choice(class_samples, min(5, len(class_samples)), replace=False)
        
        for j, sample in enumerate(samples):
            img_path = os.path.join(output_dir, sample)
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                plt.subplot(num_classes, 5, i * 5 + j + 1)
                plt.imshow(img)
                plt.title(cls)
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process based on dataset
    if args.dataset.lower() == 'food-101':
        process_food101(
            args.data_dir, 
            args.output, 
            args.split, 
            args.img_size, 
            args.num_classes, 
            args.max_samples,
            args.workers
        )
    elif args.dataset.lower() in ['uecfood', 'uec-food']:
        process_uecfood(
            args.data_dir, 
            args.output, 
            args.split, 
            args.img_size, 
            args.num_classes, 
            args.max_samples,
            args.workers
        )
    else:
        print(f"Dataset {args.dataset} not supported")
        return
    
    # Create sample images for demo
    create_sample_images(args.output)
    
    # Create visualization
    create_visualization(args.output)
    
    print(f"Preprocessing complete. Data saved to {args.output}")

if __name__ == "__main__":
    main()
