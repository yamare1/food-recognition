# Multimodal Food Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning-based system that recognizes food in images, estimates portion sizes, provides nutritional information, and answers natural language queries about food.

![Demo Image](docs/images/demo.png)

## Features

- **Food Classification**: Identifies different types of food in images using deep learning
- **Portion Estimation**: Calculates approximate portion size and weight
- **Nutritional Analysis**: Provides comprehensive nutritional information
- **Natural Language Interface**: Answer questions about the food through text queries
- **Interactive Web Interface**: User-friendly Streamlit application

## Project Structure

```
food-recognition/
├── data/                 # Data directory
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   ├── nutrition.db      # SQLite nutrition database
│   └── samples/          # Sample images for demo
├── models/               # Model implementations
│   ├── food_classifier.py
│   └── portion_estimator.py
├── utils/                # Utility functions
│   ├── nutrition_db.py   
│   └── text_processor.py
├── ui/                   # Streamlit UI
│   └── app.py
├── notebooks/            # Development notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── tests/                # Unit tests
├── main.py               # Main script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/food-recognition.git
   cd food-recognition
   ```

2. Create a virtual environment
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. Download the Food-101 dataset or another food image dataset
   ```bash
   mkdir -p data/raw
   cd data/raw
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```

2. Preprocess the dataset
   ```bash
   python scripts/preprocess_data.py --dataset food-101 --output data/processed
   ```

### Training Models

1. Train the food classification model
   ```bash
   python scripts/train_classifier.py --data data/processed/food-101 --epochs 15 --batch-size 32
   ```

2. Train or fine-tune the depth estimation model (optional)
   ```bash
   python scripts/train_depth_model.py --data data/processed/depth_dataset --epochs 10
   ```

### Running the Application

1. Run the Streamlit web application
   ```bash
   streamlit run ui/app.py
   ```

2. Alternatively, use the command-line interface
   ```bash
   python main.py --image path/to/food_image.jpg --query "How many calories are in this food?"
   ```

## Usage Examples

### Python API

```python
from food_recognition import FoodRecognitionSystem

# Initialize the system
system = FoodRecognitionSystem()

# Analyze an image
results = system.analyze_image("path/to/image.jpg")

# Print nutrition information
print(f"Food: {results['food_class']}")
print(f"Calories: {results['nutrition_data']['calories']} kcal")

# Process a query
response = system.process_query("Is this food healthy?", "path/to/image.jpg")
print(response['answer'])
```

### Web Interface

The Streamlit web application provides an intuitive interface:

1. Upload a food image or select a demo image
2. View the classification results and nutritional information
3. Ask questions about the detected food
4. Explore the food database

## Model Details

### Food Classifier

- Architecture: EfficientNet-B0 with custom classification head
- Training: Transfer learning on Food-101 dataset
- Input: 224x224 RGB images
- Output: Food categories with confidence scores

### Portion Estimator

- Uses depth estimation and reference object detection
- Estimates food volume based on spatial dimensions
- Converts volume to weight using food density approximations

## Extending the Project

### Adding New Food Categories

1. Collect images for the new categories
2. Add the categories to the training dataset
3. Retrain the classification model or use incremental learning
4. Update the nutrition database with information for new foods

### Improving Portion Estimation

1. Collect or create a dataset with annotated portion sizes
2. Train a more accurate portion estimation model
3. Implement more sophisticated 3D reconstruction techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Food-101 dataset: Bossard, Lukas, et al. "Food-101–mining discriminative components with random forests." European Conference on Computer Vision. Springer, 2014.
- Nutrition data: USDA National Nutrient Database
- PyTorch and torchvision
- Streamlit for the web interface
