# Deployment Guide: Food Recognition System

This guide provides instructions for deploying the Food Recognition System in different environments.

## Table of Contents

1. [Local Development Deployment](#local-development-deployment)
2. [Production Deployment](#production-deployment)
   - [Docker Container](#docker-container)
   - [Cloud Deployment](#cloud-deployment)
3. [Mobile Integration](#mobile-integration)
4. [Model Optimization](#model-optimization)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Development Deployment

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/food-recognition.git
   cd food-recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models:
   ```bash
   mkdir -p models
   # Download models (You can create a script for this)
   python scripts/download_models.py
   ```

5. Initialize the database:
   ```bash
   python -c "from utils.nutrition_db import NutritionDatabase; db = NutritionDatabase(); db.populate_demo_data()"
   ```

6. Run the Streamlit web application:
   ```bash
   streamlit run ui/app.py
   ```

The application should now be accessible at http://localhost:8501.

## Production Deployment

### Docker Container

#### Prerequisites

- Docker
- Docker Compose (optional)

#### Creating a Docker Image

1. Create a Dockerfile in the project root:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Initialize the database
RUN python -c "from utils.nutrition_db import NutritionDatabase; db = NutritionDatabase(); db.populate_demo_data()"

# Expose the port
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build the Docker image:
   ```bash
   docker build -t food-recognition:latest .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 food-recognition:latest
   ```

#### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  food-recognition:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Start the service:
```bash
docker-compose up -d
```

### Cloud Deployment

#### AWS Deployment

1. **EC2 Instance**:
   - Launch an EC2 instance with Ubuntu Server
   - Install Docker
   - Deploy using the Docker container method above
   - Configure security groups to allow traffic on port 8501

2. **Using AWS Elastic Beanstalk**:
   - Create a `Dockerrun.aws.json` file:
   ```json
   {
     "AWSEBDockerrunVersion": "1",
     "Image": {
       "Name": "your-ecr-repo/food-recognition:latest",
       "Update": "true"
     },
     "Ports": [
       {
         "ContainerPort": 8501,
         "HostPort": 8501
       }
     ]
   }
   ```
   - Create an Elastic Beanstalk environment
   - Upload the Dockerrun.aws.json file

#### Google Cloud Platform

1. **Deploy to Google Cloud Run**:
   ```bash
   # Build the container
   gcloud builds submit --tag gcr.io/PROJECT_ID/food-recognition
   
   # Deploy to Cloud Run
   gcloud run deploy food-recognition \
     --image gcr.io/PROJECT_ID/food-recognition \
     --platform managed \
     --allow-unauthenticated \
     --region us-central1
   ```

#### Azure Deployment

1. **Azure Container Instances**:
   ```bash
   # Create a resource group
   az group create --name food-recognition-group --location eastus
   
   # Create a container instance
   az container create \
     --resource-group food-recognition-group \
     --name food-recognition \
     --image your-registry/food-recognition:latest \
     --ports 8501 \
     --dns-name-label food-recognition \
     --location eastus
   ```

## Mobile Integration

### Option 1: API-Based Integration

1. Create a FastAPI service:
   ```python
   # api/main.py
   from fastapi import FastAPI, File, UploadFile
   import uvicorn
   from models.food_classifier import FoodClassifier
   from models.portion_estimator import PortionEstimator
   from utils.nutrition_db import NutritionDatabase, NutritionAPI
   from utils.text_processor import FoodQuerySystem
   import tempfile
   import os
   
   app = FastAPI(title="Food Recognition API")
   
   # Initialize models
   classifier = FoodClassifier(num_classes=101, model_path="models/food_classifier.pth")
   portion_estimator = PortionEstimator()
   nutrition_db = NutritionDatabase()
   nutrition_api = NutritionAPI(local_db=nutrition_db)
   query_system = FoodQuerySystem()
   
   @app.post("/analyze")
   async def analyze_food(file: UploadFile = File(...)):
       # Save the uploaded file temporarily
       with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
           temp.write(await file.read())
           temp_path = temp.name
       
       try:
           # Classify food
           class_indices, class_probs = classifier.predict(temp_path)
           top_class_idx = int(class_indices[0])
           food_classes = ["pizza", "pasta", "salad", "apple", "banana"]  # Replace with your classes
           food_class = food_classes[top_class_idx % len(food_classes)]
           
           # Estimate portion
           portion_data = portion_estimator.estimate_portion(
               image_path=temp_path,
               food_class=food_class
           )
           
           # Get nutrition
           nutrition_data = nutrition_api.get_nutrition(
               food_name=food_class,
               weight_grams=portion_data['weight_g']
           )
           
           # Clean up temp file
           os.unlink(temp_path)
           
           return {
               "food_class": food_class,
               "confidence": float(class_probs[0]),
               "portion_data": portion_data,
               "nutrition_data": nutrition_data
           }
       except Exception as e:
           # Clean up temp file
           os.unlink(temp_path)
           return {"error": str(e)}
   
   @app.post("/query")
   async def process_query(food_class: str, weight: float, query: str):
       response = query_system.process_query(
           query=query,
           detected_food=food_class,
           estimated_weight=weight
       )
       return response
   
   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

2. Start the API server:
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Mobile apps can then call these endpoints to analyze food images and process queries.

### Option 2: Model Conversion for On-Device Inference

1. Convert PyTorch models to TensorFlow Lite or Core ML:
   ```python
   # scripts/convert_model.py
   import torch
   from models.food_classifier import FoodClassifier
   
   # Load PyTorch model
   model = FoodClassifier(num_classes=101, model_path="models/food_classifier.pth")
   model.eval()
   
   # Example input
   example = torch.rand(1, 3, 224, 224)
   
   # Export to ONNX
   torch.onnx.export(
       model.model,
       example,
       "models/food_classifier.onnx",
       opset_version=11,
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
   )
   
   print("Model exported to ONNX format")
   ```

2. Convert ONNX to TensorFlow Lite:
   ```bash
   # Install onnx-tf converter
   pip install onnx-tf
   
   # Convert
   python -c "import onnx; import onnx_tf; model = onnx.load('models/food_classifier.onnx'); onnx_tf.backend.prepare(model).export_graph('models/food_classifier_tf'); print('Converted to TensorFlow')"
   
   # Convert to TF Lite
   python -c "import tensorflow as tf; converter = tf.lite.TFLiteConverter.from_saved_model('models/food_classifier_tf'); tflite_model = converter.convert(); open('models/food_classifier.tflite', 'wb').write(tflite_model); print('Converted to TF Lite')"
   ```

## Model Optimization

### Quantization

```python
# Quantize PyTorch model
import torch
from models.food_classifier import FoodClassifier

# Load model
model = FoodClassifier(num_classes=101, model_path="models/food_classifier.pth")
model.eval()

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model.model,  # Replace with your actual model attribute
    {torch.nn.Linear, torch.nn.Conv2d},  # Specify which layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "models/food_classifier_quantized.pth")
print("Model quantized and saved")
```

### Pruning

```python
# Prune PyTorch model
import torch
import torch.nn.utils.prune as prune
from models.food_classifier import FoodClassifier

# Load model
model = FoodClassifier(num_classes=101, model_path="models/food_classifier.pth")
model.eval()

# Prune 20% of connections in all linear layers
for name, module in model.model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
        prune.remove(module, 'weight')  # Make pruning permanent

# Save pruned model
torch.save(model.model.state_dict(), "models/food_classifier_pruned.pth")
print("Model pruned and saved")
```

## Monitoring and Maintenance

### Setting Up Monitoring

1. **Prometheus + Grafana Setup**:

   Create a `prometheus.yml` configuration:
   ```yaml
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'food_recognition'
       static_configs:
         - targets: ['localhost:8000']  # Assuming metrics endpoint is here
   ```

2. **Add metrics to the API**:
   ```python
   from prometheus_client import Counter, Histogram, start_http_server
   import time
   
   # Define metrics
   REQUESTS = Counter('food_recognition_requests_total', 'Total number of requests')
   LATENCY = Histogram('food_recognition_request_latency_seconds', 'Request latency in seconds')
   
   # Start metrics server
   start_http_server(8000)
   
   # Wrap API endpoints with metrics
   @app.post("/analyze")
   async def analyze_food(file: UploadFile = File(...)):
       REQUESTS.inc()
       start_time = time.time()
       
       # Existing function logic
       
       LATENCY.observe(time.time() - start_time)
       return result
   ```

### Continuous Integration/Deployment

Create a GitHub Actions workflow:

```yaml
# .github/workflows/deploy.yml
name: Deploy Food Recognition System

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t food-recognition:latest .
      - name: Login to Docker Hub
        uses: docker/login-action@v1
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
      - name: Push Docker image
        run: |
          docker tag food-recognition:latest ${{ secrets.DOCKER_HUB_USERNAME }}/food-recognition:latest
          docker push ${{ secrets.DOCKER_HUB_USERNAME }}/food-recognition:latest
```

### Automated Model Retraining

Create a script for automated retraining:

```python
# scripts/retrain_model.py
import argparse
import datetime
import os
from scripts.train_classifier import train_model, create_model, evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Retrain food classification model')
    parser.add_argument('--data', type=str, default='data/new_samples', help='Directory with new data')
    parser.add_argument('--model', type=str, default='models/food_classifier.pth', help='Path to existing model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of fine-tuning epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Generate timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"models/food_classifier_{timestamp}.pth"
    
    # Load and fine-tune model
    print(f"Fine-tuning model {args.model} with new data from {args.data}")
    print(f"New model will be saved to {new_model_path}")
    
    # Implement fine-tuning logic
    # ...
    
    print("Retraining complete")

if __name__ == "__main__":
    main()
```

Schedule this script to run periodically with new data.

## Backup and Recovery

### Database Backup Script

```bash
#!/bin/bash
# backup_db.sh

# Set backup directory
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_FILE="data/nutrition.db"
BACKUP_FILE="${BACKUP_DIR}/nutrition_${TIMESTAMP}.db"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create backup
cp "$DB_FILE" "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"

echo "Backup created: ${BACKUP_FILE}.gz"

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "nutrition_*.db.gz" -type f -mtime +30 -delete
```

Make the script executable and set up a cron job:
```bash
chmod +x backup_db.sh
crontab -e
# Add this line to run daily at 2 AM:
# 0 2 * * * /path/to/food-recognition/backup_db.sh
```

### Model Versioning

Create a model registry script:

```python
# utils/model_registry.py
import os
import json
import datetime
import shutil

class ModelRegistry:
    def __init__(self, registry_dir="models/registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": [],
                "current": None
            }
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, metrics, description=""):
        # Copy model to registry
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        model_version = f"{model_name.split('.')[0]}_{timestamp}.pth"
        registry_path = os.path.join(self.registry_dir, model_version)
        
        # Copy the model file
        shutil.copy(model_path, registry_path)
        
        # Register the model
        model_info = {
            "version": model_version,
            "path": registry_path,
            "timestamp": timestamp,
            "metrics": metrics,
            "description": description
        }
        
        self.registry["models"].append(model_info)
        self._save_registry()
        return model_info
    
    def set_current_model(self, version):
        # Find the model in registry
        for model in self.registry["models"]:
            if model["version"] == version:
                self.registry["current"] = version
                self._save_registry()
                return True
        return False
    
    def get_current_model(self):
        if not self.registry["current"]:
            return None
        
        for model in self.registry["models"]:
            if model["version"] == self.registry["current"]:
                return model
        return None
    
    def list_models(self):
        return self.registry["models"]
```

Usage example:
```python
# Register a new model
registry = ModelRegistry()
registry.register_model(
    model_path="models/food_classifier.pth",
    metrics={"accuracy": 0.92, "loss": 0.24},
    description="EfficientNet-B0 model trained on Food-101 dataset"
)

# Set as current model
registry.set_current_model("food_classifier_20220315_120000.pth")

# Get current model info
current_model = registry.get_current_model()
print(f"Using model {current_model['version']} with accuracy {current_model['metrics']['accuracy']}")
```

This deployment guide should help you set up and maintain the Food Recognition System in various environments. Remember to adapt these instructions to your specific setup and requirements.
