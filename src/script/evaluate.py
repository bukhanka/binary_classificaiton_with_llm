import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

from src.utils.dataset_prep import DatasetPreparation
from src.utils.feature_extractor import FeatureExtractor
from src.utils.logger import system_logger
from sklearn.linear_model import LogisticRegression
from langchain_openai import ChatOpenAI

class ModelPredictor:
    def __init__(self, method='ml'):
        """
        Initialize predictor with chosen method
        
        Args:
            method (str): Prediction method ('ml' or 'llm')
        """
        self.method = method
        self.ml_classifier = None
        self.llm_classifier = None
    
    def train(self, features, labels):
        """
        Train the predictor
        
        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): Corresponding labels
        """
        if self.method == 'ml':
            # Train Logistic Regression
            self.ml_classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.ml_classifier.fit(features, labels)
        elif self.method == 'llm':
            # Initialize LLM for classification
            self.llm_classifier = ChatOpenAI(
                model='gpt-4o', 
                temperature=0.2
            )
    
    def predict(self, features, texts=None):
        """
        Predict labels
        
        Args:
            features (np.ndarray): Feature vectors
            texts (List[str], optional): Original texts for LLM method
        
        Returns:
            np.ndarray: Predicted labels
        """
        if self.method == 'ml':
            return self.ml_classifier.predict(features)
        
        elif self.method == 'llm':
            if texts is None:
                raise ValueError("Texts required for LLM prediction")
            
            predictions = []
            for text in texts:
                # LLM-based classification prompt
                response = self.llm_classifier.invoke(
                    f"""Classify the following text as relevant (1) or irrelevant (0):
                    Text: {text}
                    
                    Reasoning steps:
                    1. Analyze semantic content
                    2. Determine relevance
                    3. Provide binary classification
                    
                    Output ONLY the number 0 or 1."""
                )
                
                # Extract numeric label
                try:
                    label = int(response.content.strip())
                    predictions.append(label)
                except ValueError:
                    # Fallback to default
                    predictions.append(0)
            
            return np.array(predictions)

def load_test_dataset(config_path='config.yaml'):
    """
    Load the test dataset from the configured path
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        pandas.DataFrame: Test dataset
    """
    dataset_prep = DatasetPreparation(config_path)
    test_df = dataset_prep.load_validation_dataset()
    return test_df

def evaluate_classification_system(test_df, method='ml'):
    """
    Comprehensive model evaluation
    
    Args:
        test_df (pandas.DataFrame): Test dataset with labeled examples
        method (str): Prediction method ('ml' or 'llm')
    
    Returns:
        dict: Performance metrics
    """
    # Validate input
    if test_df is None or len(test_df) == 0:
        system_logger.error("Empty test dataset")
        return {}
    
    # Ensure 'text' and 'label' columns exist
    if 'text' not in test_df.columns or 'label' not in test_df.columns:
        system_logger.error("Missing required columns in test dataset")
        return {}
    
    # Feature Extraction
    feature_extractor = FeatureExtractor()
    
    # Prepare Training Data
    train_data = feature_extractor.prepare_features(test_df, 'text')
    
    # Initialize and train predictor
    predictor = ModelPredictor(method=method)
    predictor.train(train_data['features'], train_data['labels'])
    
    # Predict labels
    test_features = feature_extractor.extract_features(test_df['text'])
    predictions = predictor.predict(
        test_features, 
        texts=test_df['text'].tolist() if method == 'llm' else None
    )
    
    # Compute Detailed Metrics
    metrics = {
        'accuracy': accuracy_score(test_df['label'], predictions),
        'precision': precision_score(test_df['label'], predictions),
        'recall': recall_score(test_df['label'], predictions),
        'f1_score': f1_score(test_df['label'], predictions),
        'confusion_matrix': confusion_matrix(test_df['label'], predictions).tolist(),
        'classification_report': classification_report(test_df['label'], predictions)
    }
    
    # Log Performance
    system_logger.log_model_performance(metrics)
    
    # Detailed Logging
    system_logger.info(f"\nDetailed Classification Report ({method} method):")
    system_logger.info(metrics['classification_report'])
    
    return metrics

def main():
    try:
        system_logger.info("Starting Comprehensive Model Evaluation")
        
        # Load Test Dataset
        test_df = load_test_dataset()
        system_logger.info(f"Test Dataset Loaded: {len(test_df)} samples")
        
        # Evaluate using ML method
        ml_metrics = evaluate_classification_system(test_df, method='ml')
        
        # Evaluate using LLM method (optional)
        try:
            llm_metrics = evaluate_classification_system(test_df, method='llm')
        except Exception as e:
            system_logger.warning(f"LLM evaluation failed: {e}")
            llm_metrics = None
        
        # Print ML Metrics
        print("\n===== ML Model Performance Metrics =====")
        for metric, value in ml_metrics.items():
            if metric != 'classification_report':
                print(f"{metric}: {value}")
        
        # Print LLM Metrics if available
        if llm_metrics:
            print("\n===== LLM Model Performance Metrics =====")
            for metric, value in llm_metrics.items():
                if metric != 'classification_report':
                    print(f"{metric}: {value}")
        
        system_logger.info("Model Evaluation Completed Successfully")
    
    except Exception as e:
        system_logger.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main() 