import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.utils.data_preprocessor import DataPreprocessor
from src.utils.feature_extractor import FeatureExtractor
from src.utils.dataset_prep import DatasetPreparation
from src.agents.classification_agent import ClassificationAgent
from src.utils.logger import system_logger

def setup_environment():
    """
    Set up the working environment and validate configurations
    """
    # Ensure required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def main():
    try:
        # Environment Setup
        setup_environment()
        system_logger.info("Classification System Initialization Started")

        # Dataset Preparation
        dataset_prep = DatasetPreparation()
        system_logger.info("Dataset Preparation Initiated")
        
        # Create Validation Dataset
        validation_df = dataset_prep.create_validation_dataset()
        system_logger.info(f"Validation Dataset Created: {len(validation_df)} samples")
        
        # Load Original Datasets
        relevant_df, irrelevant_df = dataset_prep.load_datasets()
        system_logger.info(f"Loaded Datasets: {len(relevant_df)} relevant, {len(irrelevant_df)} irrelevant")
        
        # Feature Extraction
        feature_extractor = FeatureExtractor()
        train_data = feature_extractor.prepare_features(
            pd.concat([relevant_df, irrelevant_df]), 
            'text'
        )
        system_logger.info("Feature Extraction Completed")
        
        # Train Classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(train_data['features'], train_data['labels'])
        system_logger.info("Classifier Training Completed")
        
        # Create Classification Agent
        classification_agent = ClassificationAgent(
            feature_extractor=feature_extractor, 
            classifier=classifier,
            relevant_df=relevant_df,
            irrelevant_df=irrelevant_df,
            validation_dataset=validation_df
        )
        
        # Create LangGraph Workflow
        graph = classification_agent.create_graph()
        system_logger.info("LangGraph Workflow Created")
        
        # Model Evaluation
        metrics = dataset_prep.evaluate_model(classification_agent, validation_df)
        system_logger.log_model_performance(metrics)
        
        # Example Inference
        system_logger.info("Starting Example Inferences")
        test_texts = validation_df['text'].head(5).tolist()
        for text in test_texts:
            result = graph.invoke({"input": text})
            system_logger.info(f"Inference Result: {result}")
        
        system_logger.info("Classification System Execution Completed Successfully")

    except Exception as e:
        system_logger.error(f"Critical Error in Classification System: {e}")
        system_logger.error(f"Error Details: {sys.exc_info()}")
        raise

if __name__ == "__main__":
    main() 