import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

class FoodDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images
            annotations_file (string): Path to the annotations file
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        
        # Create class mappings
        self.classes = sorted(self.annotations['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.annotations.iloc[idx, 1]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FoodClassifier:
    def __init__(self, num_classes, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model(num_classes)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
            
        self.model.to(self.device)
        
    def _create_model(self, num_classes):
        """Create a pre-trained model and modify for food classification"""
        model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the last fully connected layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        return model
    
    def train(self, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001):
        """Train the food classification model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_dataloader.dataset)
            train_acc = correct / total
            
            # Validation loop
            val_loss, val_acc = self.evaluate(val_dataloader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/best_food_classifier.pth')
                print("Saved best model checkpoint")
    
    def evaluate(self, dataloader, criterion=None):
        """Evaluate the model on the given dataloader"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return running_loss / len(dataloader.dataset), correct / total
    
    def predict(self, image_path):
        """Predict food category for a single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_p, top_class = probabilities.topk(5, dim=1)
            
        return top_class.squeeze().cpu().numpy(), top_p.squeeze().cpu().numpy()


def prepare_data_loaders(data_dir, annotations_file, batch_size=32, train_ratio=0.8):
    """Prepare train and validation dataloaders"""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    df = pd.read_csv(annotations_file)
    train_df = df.sample(frac=train_ratio, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df.to_csv('data/train_annotations.csv', index=False)
    val_df.to_csv('data/val_annotations.csv', index=False)
    
    train_dataset = FoodDataset(data_dir, 'data/train_annotations.csv', transform=train_transform)
    val_dataset = FoodDataset(data_dir, 'data/val_annotations.csv', transform=val_transform)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_dataloader, val_dataloader, train_dataset.classes


if __name__ == "__main__":
    # Example usage
    data_dir = "data/food-101/images"
    annotations_file = "data/food-101/meta/labels.csv"
    
    # Create dataloaders
    train_loader, val_loader, classes = prepare_data_loaders(data_dir, annotations_file)
    
    # Initialize and train model
    food_classifier = FoodClassifier(num_classes=len(classes))
    food_classifier.train(train_loader, val_loader, num_epochs=15)
    
    # Evaluate on validation set
    val_loss, val_acc = food_classifier.evaluate(val_loader)
    print(f"Final validation accuracy: {val_acc:.4f}")
