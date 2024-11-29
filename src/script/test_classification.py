import pandas as pd
import sys
import os
import logging
from typing import List, Dict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.classification_agent import ClassificationAgent
from src.utils.dataset_prep import DatasetPreparation
from src.utils.feature_extractor import FeatureExtractor
from sklearn.linear_model import LogisticRegression
from langchain_openai import ChatOpenAI

def test_classification(
    dataset_path: str, 
    log_file: str = 'classification_test_log.txt',
    num_samples: int = None
) -> pd.DataFrame:
    """
    Test classification on a specific dataset
    
    Args:
        dataset_path (str): Path to the dataset
        log_file (str): Path to log file
        num_samples (int, optional): Number of samples to test
    
    Returns:
        DataFrame with classification results
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Read the dataset (handle both CSV and XLSX)
        if dataset_path.endswith('.xlsx'):
            df = pd.read_excel(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
        
        # Ensure 'post' column exists
        if 'post' not in df.columns:
            # Try to find text column
            text_columns = df.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                df = df.rename(columns={text_columns[0]: 'post'})
            else:
                raise ValueError("No text column found in the dataset")
        
        # Add an ID column if not exists
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        
        # Limit samples if specified
        if num_samples:
            df = df.head(num_samples)
        
        # Prepare feature extractor and classifier
        feature_extractor = FeatureExtractor()
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        
        # Create dataset preparation instance
        dataset_prep = DatasetPreparation()
        
        # Create validation dataset
        validation_df = dataset_prep.create_validation_dataset()
        
        # Create Classification Agent
        classification_agent = ClassificationAgent(
            feature_extractor=feature_extractor, 
            classifier=classifier,
            relevant_df=pd.DataFrame(columns=['post', 'label']),
            irrelevant_df=pd.DataFrame(columns=['post', 'label']),
            validation_dataset=validation_df.head(100)
        )
        
        # Classify each text
        results = []
        classification_details = []
        
        for idx, row in df.iterrows():
            try:
                # Perform classification
                result = classification_agent.initial_classification({'input': row['post']})
                
                # Prepare detailed logging
                detail_entry = {
                    'row_id': idx,
                    'text': row['post'],
                    'classification': result['final_classification'],
                    'confidence': result['confidence_score'],
                    'intermediate_steps': result['intermediate_steps'],
                    'retrieved_examples': result.get('retrieved_examples', []),
                    'label_distribution': result.get('label_distribution', {})
                }
                classification_details.append(detail_entry)
                
                results.append({
                    'id': row['id'],
                    'post': row['post'],
                    'category': result['final_classification']
                })
            
            except Exception as row_error:
                logger.error(f"Error classifying row {idx}: {row_error}")
                results.append({
                    'id': row['id'],
                    'post': row['post'],
                    'category': 0  # Default to irrelevant
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = 'test_classification_results.csv'
        results_df.to_csv(output_path, index=False)
        
        # Save detailed classification details
        details_path = 'test_classification_details.json'
        import json
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(classification_details, f, ensure_ascii=False, indent=2)
        
        # Analyze classification distribution
        class_distribution = results_df['category'].value_counts(normalize=True)
        
        logger.info("Classification Distribution:")
        for label, proportion in class_distribution.items():
            logger.info(f"  Label {label}: {proportion*100:.2f}%")
        
        print(f"Test complete. Results saved to {output_path}")
        print(f"Detailed classification details saved to {details_path}")
        
        return results_df
    
    except Exception as e:
        logger.error(f"Critical error during test: {e}")
        raise

# If script is run directly
if __name__ == "__main__":
    dataset_path = '/home/dukhanin/soc_hack/docs/Хакатон_2024/Примеры нерелевантных постов.xlsx'
    test_classification(dataset_path, num_samples=50) 