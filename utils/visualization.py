import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_prediction(image_path, food_class, confidence, nutrition_data):
    """Visualize food prediction with nutrition information"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f"Detected: {food_class.capitalize()}\nConfidence: {confidence*100:.1f}%")
    ax1.axis('off')
    
    # Display nutrition
    ax2.axis('off')
    nutrition_text = f"Nutrition Facts\n\nCalories: {nutrition_data['calories']:.0f} kcal\n"
    nutrition_text += f"Protein: {nutrition_data['protein_g']:.1f}g\n"
    nutrition_text += f"Fat: {nutrition_data['fat_g']:.1f}g\n"
    nutrition_text += f"Carbs: {nutrition_data['carbs_g']:.1f}g"
    
    ax2.text(0.1, 0.5, nutrition_text, fontsize=12)
    
    plt.tight_layout()
    return fig
