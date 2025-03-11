#!/usr/bin/env python3
"""
Training script for the Food Recognition System models
This script trains the food classification model.
"""

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

class FoodDataset(Dataset):
    """Food image dataset loader"""
    
    def __init__(self, annotations_file, data_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the annotations CSV file
            data_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        
        # Get class mapping
        self.classes = sorted(self.annotations['class'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.annotations.iloc[idx, 1]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def parse_args():
    parser = argparse.ArgumentParser(description='Train food classification model')
    parser.add_argument('--data', type=str, default='data/processed', help='Directory with processed dataset')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                        choices=['efficientnet_b0', 'resnet50', 'mobilenet_v2'], 
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--resume', type=str, default='', help='Path to resume training from checkpoint')
    return parser.parse_args()

def create_model(backbone, num_classes, pretrained=True):
    """Create model with specified backbone"""
    print(f"Creating model with {backbone} backbone...")
    
    if backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        # Modify classifier head
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Modify classifier head
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif backbone == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        # Modify classifier head
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs, model_dir, backbone):
    """Train the model"""
    
    # Create directory for model checkpoints
    os.makedirs(model_dir, exist_ok=True)
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_samples += inputs.size(0)
        
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_corrects.double() / train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += inputs.size(0)
        
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects.double() / val_samples
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Print epoch results
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Record history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        # Save checkpoint if best model so far
        if epoch_val_acc > history['best_val_acc']:
            history['best_val_acc'] = epoch_val_acc.item()
            history['best_epoch'] = epoch
            
            # Save model checkpoint
            checkpoint_path = os.path.join(model_dir, f"best_{backbone}_food_classifier.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(model_dir, f"final_{backbone}_food_classifier.pth")
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': epoch_val_acc,
        'val_loss': epoch_val_loss,
    }, final_path)
    
    # Save training history
    history_path = os.path.join(model_dir, f"{backbone}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']+1}")
    
    return history

def evaluate_model(model, val_loader, criterion, device, class_names):
    """Evaluate model performance on validation set"""
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_samples = 0
    
    # For detailed metrics
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            val_samples += inputs.size(0)
            
            # Save predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_loss = val_loss / val_samples
    val_acc = val_corrects.double() / val_samples
    
    print(f"\nFinal Evaluation:")
    print(f"Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")
    
    # Get classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        digits=3, 
        output_dict=True
    )
    
    # Convert to DataFrame for easier viewing
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc.item(),
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_training_history(history, output_dir):
    """Plot training history graphs"""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], '.2f'),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation and preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.img_size + 32),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    annotations_file = os.path.join(args.data, 'annotations.csv')
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Load datasets
    annotations = pd.read_csv(annotations_file)
    train_df = annotations[annotations['split'] == 'train']
    val_df = annotations[annotations['split'] == 'val']
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = FoodDataset(
        annotations_file=annotations_file,
        data_dir=args.data,
        transform=data_transforms['train']
    )
    
    val_dataset = FoodDataset(
        annotations_file=annotations_file,
        data_dir=args.data,
        transform=data_transforms['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    if args.resume:
        # Resume from checkpoint
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model = create_model(args.backbone, num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']+1}")
    else:
        # Create new model
        model = create_model(args.backbone, num_classes, pretrained=args.pretrained)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        model_dir=args.model_dir,
        backbone=args.backbone
    )
    
    # Plot training results
    plot_training_history(history, args.model_dir)
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.model_dir, f"best_{args.backbone}_food_classifier.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate model
    eval_results = evaluate_model(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=eval_results['confusion_matrix'],
        class_names=class_names,
        output_dir=args.model_dir
    )
    
    # Save model in a format ready for inference
    torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.backbone}_food_classifier.pth"))
    
    # Save class names
    with open(os.path.join(args.model_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"Training and evaluation complete. Model saved to {args.model_dir}")

if __name__ == "__main__":
    main()
