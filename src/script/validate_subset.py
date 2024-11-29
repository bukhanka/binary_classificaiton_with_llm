import pandas as pd
import sys
import os
import time
import json
from typing import List, Dict
import logging
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.classification_agent import ClassificationAgent
from src.utils.dataset_prep import DatasetPreparation
from src.utils.feature_extractor import FeatureExtractor
from sklearn.linear_model import LogisticRegression

def validate_subset(
    dataset_path: str, 
    num_rows: int = 50, 
    log_file: str = 'validation_log.txt'
) -> pd.DataFrame:
    """
    Validate a subset of the dataset with detailed logging and timing
    
    Args:
        dataset_path (str): Path to the input CSV
        num_rows (int): Number of rows to validate
        log_file (str): Path to log file
    
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
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare logging
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write(f"Validation Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Dataset: {dataset_path}\n")
            log.write(f"Rows to validate: {num_rows}\n\n")
            
            try:
                # Read the dataset
                df = pd.read_csv(dataset_path)
                
                # Take first N rows
                subset_df = df.head(num_rows).copy()
                log.write(f"Subset DataFrame Shape: {subset_df.shape}\n")
                
                # Prepare feature extractor and classifier
                feature_extractor = FeatureExtractor()
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                
                # Create dataset preparation instance
                dataset_prep = DatasetPreparation()
                
                # Create validation dataset (will not recreate if exists)
                validation_df = dataset_prep.create_validation_dataset()
                log.write(f"Validation Dataset Shape: {validation_df.shape}\n\n")
                
                # Create Classification Agent with limited embedding
                classification_agent = ClassificationAgent(
                    feature_extractor=feature_extractor, 
                    classifier=classifier,
                    relevant_df=pd.DataFrame(columns=['post', 'label']),  # Empty DataFrame
                    irrelevant_df=pd.DataFrame(columns=['post', 'label']),  # Empty DataFrame
                    validation_dataset=validation_df.head(100)  # Limit validation dataset
                )
                
                # Classify each text
                results = []
                classification_times = []
                classification_details = []
                
                for idx, row in subset_df.iterrows():
                    try:
                        # Time individual classification
                        classification_start = time.time()
                        
                        try:
                            result = classification_agent.initial_classification({'input': row['post']})
                        except (RateLimitError, APIConnectionError) as api_error:
                            logger.warning(f"API error on row {idx}: {api_error}")
                            # Wait and retry
                            time.sleep(5)  # Wait 5 seconds
                            result = classification_agent.initial_classification({'input': row['post']})
                        
                        classification_end = time.time()
                        
                        # Log individual classification details
                        classification_time = classification_end - classification_start
                        classification_times.append(classification_time)
                        
                        # Prepare detailed logging
                        detail_entry = {
                            'row_id': idx,
                            'text': row['post'],
                            'classification': result['final_classification'],
                            'confidence': result['confidence_score'],
                            'classification_time': classification_time,
                            'intermediate_steps': result['intermediate_steps'],
                            'retrieved_examples': [
                                {
                                    'text': ex['text'], 
                                    'label': ex.get('label', 'N/A'), 
                                    'distance': ex.get('distance', 'N/A')
                                } 
                                for ex in result.get('retrieved_examples', [])
                            ],
                            'label_distribution': result.get('label_distribution', {})
                        }
                        classification_details.append(detail_entry)
                        
                        # Write detailed log
                        log.write(f"Row {idx}:\n")
                        log.write(f"  Text: {row['post'][:200]}...\n")
                        log.write(f"  Classification: {result['final_classification']}\n")
                        log.write(f"  Confidence: {result['confidence_score']:.2f}\n")
                        log.write(f"  Classification Time: {classification_time:.4f} seconds\n")
                        
                        # Log intermediate steps
                        log.write("  Intermediate Steps:\n")
                        for step in result['intermediate_steps']:
                            log.write(f"    - {step}\n")
                        
                        # Log retrieved examples
                        log.write("  Retrieved Examples:\n")
                        for ex in result.get('retrieved_examples', []):
                            log.write(f"    - Text: {ex['text'][:100]}...\n")
                            log.write(f"      Label: {ex.get('label', 'N/A')}\n")
                            log.write(f"      Distance: {ex.get('distance', 'N/A')}\n")
                        
                        log.write("\n")
                        
                        results.append({
                            'id': row['id'],
                            'post': row['post'],
                            'category': result['final_classification']
                        })
                    except Exception as row_error:
                        logger.error(f"Error classifying row {idx}: {row_error}")
                        # Add a default classification in case of error
                        results.append({
                            'id': row['id'],
                            'post': row['post'],
                            'category': 0  # Default to irrelevant
                        })
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                
                # Save results
                output_path = 'classification_results.csv'
                results_df.to_csv(output_path, index=False)
                
                # Save detailed classification details
                details_path = 'classification_details.json'
                with open(details_path, 'w', encoding='utf-8') as f:
                    json.dump(classification_details, f, ensure_ascii=False, indent=2)
                
                # Calculate total time and performance metrics
                end_time = time.time()
                total_time = end_time - start_time
                
                # Analyze classification distribution
                class_distribution = results_df['category'].value_counts(normalize=True)
                
                log.write("Performance Summary:\n")
                log.write(f"Total Validation Time: {total_time:.2f} seconds\n")
                log.write(f"Average Classification Time: {sum(classification_times)/len(classification_times):.4f} seconds\n")
                log.write(f"Classification Distribution:\n")
                for label, proportion in class_distribution.items():
                    log.write(f"  Label {label}: {proportion*100:.2f}%\n")
                
                log.write(f"\nResults saved to: {output_path}\n")
                log.write(f"Detailed classification details saved to: {details_path}\n")
                
                print(f"Validation complete. Check {log_file} for details.")
                return results_df
            
            except Exception as e:
                log.write(f"Error during validation: {str(e)}\n")
                raise
    
    except Exception as e:
        logger.error(f"Critical error during validation: {e}")
        raise

# If script is run directly
if __name__ == "__main__":
    dataset_path = '/home/dukhanin/soc_hack/docs/Хакатон_2024/dataset_select_hackathon_26_11.csv'
    validate_subset(dataset_path, num_rows=50) 