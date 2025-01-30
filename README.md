# Chest Cancer Classification using MLOps

## Project Overview
This project focuses on classifying chest cancer using Machine Learning Operations (MLOps) principles. The workflow involves data ingestion, model training, evaluation, and deployment using modern MLOps tools such as MLflow, DVC, Flask, and Docker.

## Project Structure
The project follows a standard MLOps directory structure:
```
project_root/
│-- config/
│   │-- config.yaml
│   │-- params.yaml
│-- artifacts/
│   │-- data_ingestion/
│   │-- prepare_base_model/
│   │-- training/
│   │-- evaluation/
│-- src/
│   │-- data_ingestion.py
│   │-- prepare_base_model.py
│   │-- model_trainer.py
│   │-- model_evaluation.py
│   │-- prediction.py
│-- notebooks/
│-- dvc.yaml
│-- Dockerfile
│-- requirements.txt
│-- app.py
│-- README.md
```

## Steps in the Project

### 1. Data Ingestion
- Downloads dataset from a remote source(google drive zip file).
- Extract the zip file and stores it in the `artifacts/data_ingestion/` directory.

### 2. Preparing Base Model
- Loads necessary configurations.
- Load a base model for training.
- Stores it in `artifacts/prepare_base_model/`.

### 3. Model Training
- Loads data and prepares the model for training.
- Trains the model and stores the results in `artifacts/training/`.

### 4. Model Evaluation
- Evaluates the trained model.
- Saves the evaluation scores in `scores.json`.
- Logs warnings for untraced functions during saving.

### 5. MLflow and DVC Integration
- Connected to MLflow using DAGsHub for tracking experiments.
- Used DVC (Data Version Control) for pipeline management.
- Below is an image showcasing the DVC pipeline:
  
  ![DVC Pipeline]((https://github.com/user-attachments/assets/75fdac71-0f13-4230-b4ad-010020b35f71.png)


### 6. Flask Prediction Pipeline
- Developed a Flask API (`app.py`) for serving model predictions.

### 7. Docker Containerization
- Created a Docker image of the project.
- The image can be deployed on any cloud or server.
- The container has been uploaded to Docker Hub and can be pulled using the command:
  ```sh
  docker pull vibhav70/chestcancer_mlops
  ```

## Running the Project
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Docker
- DVC
- MLflow
- Flask
- Required Python libraries (`requirements.txt`)

### Steps to Run
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Vibhav70/chest-cancer-classification-using-mlflow.git
   cd chest-cancer-classification-using-mlflow
   ```
2. **Create and Activate Environment:**
   ```sh
   python -m venv venv
   cd venv
   cd Scripts
   activate
   cd..
   cd..
   ```   
4. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
5. **Run the Pipeline Using DVC:**
   ```sh
   dvc repro
   ```
6. **Start Flask Application:**
   ```sh
   python app.py
   ```
### Alternate Way
1. **Run with Docker:**
   ```sh
   docker build -t chest_cancer_classifier .
   docker run -p 5000:5000 chest_cancer_classifier
   ```
2. **Pull Docker Image from Docker Hub:**
   ```sh
   docker pull your_dockerhub_username/chest_cancer_classifier
   ```

## Results and Observations
- Successfully implemented MLOps principles for chest cancer classification.
- Automated model training and evaluation.
- Integrated experiment tracking and data version control.
- Developed an API for easy access to predictions.
- Packaged the model in a portable Docker container for easy deployment.

## Future Improvements
- Enhance model performance with hyperparameter tuning.
- Deploy the API on cloud platforms (AWS, GCP, Azure).
- Implement continuous monitoring and feedback loops.

---
This README provides an overview of the project, detailing the MLOps pipeline and necessary steps for execution.

