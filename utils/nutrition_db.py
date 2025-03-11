import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class DepthEstimator:
    """Estimates depth in an image to help with portion size estimation"""
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self):
        """Load a pre-trained MiDaS model for depth estimation"""
        # Using a simpler model for example purposes
        # In a real implementation, you would load MiDaS or another depth estimation model
        model = models.segmentation.fcn_resnet50(pretrained=True)
        # Modify last layer for depth estimation (simplified)
        model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=1)
        return model
    
    def predict_depth(self, image_path):
        """Predict depth map for an image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            depth_map = output.squeeze().cpu().numpy()
            
        # Normalize depth map for visualization
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)
        
        return depth_map


class PortionEstimator:
    """Estimates food portion sizes using reference objects and depth information"""
    
    def __init__(self, depth_model_path=None):
        self.depth_estimator = DepthEstimator(model_path=depth_model_path)
        self.reference_objects = {
            'credit_card': {'width': 8.56, 'height': 5.398},  # cm
            'bottle_cap': {'diameter': 3.0},  # cm
            'smartphone': {'width': 7.0, 'height': 14.0}  # approximate average in cm
        }
        
    def detect_reference_object(self, image_path):
        """Detect if there's a reference object in the image"""
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For simplicity, we'll just assume a credit card might be present
        # In a real implementation, you would use object detection models
        
        # Use Hough transform to detect rectangles that might be credit cards
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_references = []
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If it's a rectangle (4 points)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # Credit card aspect ratio is approximately 1.586
                if 1.4 < aspect_ratio < 1.7 and w > 100:
                    potential_references.append({
                        'type': 'credit_card',
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,  # Placeholder confidence
                        'pixels_per_cm': w / self.reference_objects['credit_card']['width']
                    })
        
        return potential_references
    
    def estimate_food_volume(self, image_path, food_bbox, reference_object=None):
        """
        Estimate food volume using depth information and reference objects
        
        Args:
            image_path: Path to the food image
            food_bbox: Bounding box of the food (x, y, w, h)
            reference_object: Detected reference object with pixel-to-cm ratio
            
        Returns:
            estimated_volume: Estimated volume in cubic centimeters (cc)
            estimated_weight: Estimated weight in grams
        """
        x, y, w, h = food_bbox
        
        # Get depth map
        depth_map = self.depth_estimator.predict_depth(image_path)
        
        # Extract depth in food region
        food_depth = depth_map[y:y+h, x:x+w]
        avg_depth = np.mean(food_depth)
        
        # If no reference object is provided, try to detect one
        if reference_object is None:
            detected_refs = self.detect_reference_object(image_path)
            if detected_refs:
                reference_object = detected_refs[0]
            else:
                # Use a default scale if no reference object is found
                # This is very approximate and would be inaccurate
                reference_object = {
                    'pixels_per_cm': 30  # Assuming 30 pixels = 1 cm
                }
        
        # Calculate physical dimensions
        width_cm = w / reference_object['pixels_per_cm']
        height_cm = h / reference_object['pixels_per_cm']
        
        # Use depth to approximate the 3D shape
        # Simplified model: treat food as a cylinder or a box
        # This is a very rough approximation and would need to be refined
        
        area_cm2 = width_cm * height_cm
        depth_factor = avg_depth / 128  # Normalize depth to a 0-2 range (very simplified)
        volume_cc = area_cm2 * depth_factor * 5  # Simplified volume estimation
        
        # Estimate weight based on density
        # Different foods have different densities, but as an approximation:
        # Water is 1g/cc, most foods range from 0.7-1.2 g/cc
        density = 0.8  # g/cc, approximated average for many foods
        weight_g = volume_cc * density
        
        return {
            'volume_cc': volume_cc,
            'weight_g': weight_g,
            'dimensions': {
                'width_cm': width_cm,
                'height_cm': height_cm,
                'depth_factor': depth_factor
            }
        }
    
    def estimate_portion(self, image_path, food_class, bbox=None):
        """
        Estimates portion size for a food item in the image
        
        Args:
            image_path: Path to the food image
            food_class: The identified food class
            bbox: Bounding box (optional, will be detected if not provided)
            
        Returns:
            portion_data: Portion information including estimated weight and volume
        """
        # Load image
        image = cv2.imread(image_path)
        
        # If bbox not provided, attempt to segment the food
        if bbox is None:
            # Simplified segmentation using thresholding
            # In a real implementation, use a proper segmentation model
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find largest contour
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                bbox = cv2.boundingRect(largest_contour)
            else:
                # Default to whole image if no contour is found
                h, w = image.shape[:2]
                bbox = (0, 0, w, h)
        
        # Detect reference objects
        reference_objects = self.detect_reference_object(image_path)
        reference_object = reference_objects[0] if reference_objects else None
        
        # Estimate food volume
        volume_data = self.estimate_food_volume(image_path, bbox, reference_object)
        
        # Adjust portion estimate based on food class
        # Different foods have different densities and typical portion sizes
        # This would be improved with a proper food database
        typical_portions = {
            'pizza': {'serving_size_g': 100, 'density': 0.9},
            'salad': {'serving_size_g': 150, 'density': 0.7},
            'pasta': {'serving_size_g': 180, 'density': 1.0},
            'steak': {'serving_size_g': 200, 'density': 1.1},
            'apple': {'serving_size_g': 150, 'density': 0.85},
            # Add more foods based on your classification model
        }
        
        if food_class in typical_portions:
            # Adjust estimate using typical food properties
            density = typical_portions[food_class]['density']
            volume_data['weight_g'] = volume_data['volume_cc'] * density
            volume_data['servings'] = volume_data['weight_g'] / typical_portions[food_class]['serving_size_g']
        else:
            # Default serving calculation
            volume_data['servings'] = volume_data['weight_g'] / 100  # Assuming 100g per serving
        
        return volume_data


if __name__ == "__main__":
    # Example usage
    estimator = PortionEstimator()
    
    # Estimate portion for a food image
    portion_data = estimator.estimate_portion(
        image_path="data/samples/pizza.jpg",
        food_class="pizza",
        bbox=None  # Will be detected automatically
    )
    
    print(f"Estimated volume: {portion_data['volume_cc']:.2f} cc")
    print(f"Estimated weight: {portion_data['weight_g']:.2f} g")
    print(f"Estimated servings: {portion_data['servings']:.2f}")
