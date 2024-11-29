import pandas as pd
import sys
import os
import time
import json
import traceback
from typing import List, Dict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.classification_agent import ClassificationAgent
from src.utils.dataset_prep import DatasetPreparation
from src.utils.feature_extractor import FeatureExtractor
from sklearn.linear_model import LogisticRegression

def validate_full_dataset(
    dataset_path: str, 
    output_path: str = 'classification_results.csv',
    log_file: str = 'full_validation_log.txt',
    batch_size: int = 100,
    resume: bool = False
) -> pd.DataFrame:
    """
    Validate full dataset with robust error handling and resumability
    
    Args:
        dataset_path (str): Path to the input CSV
        output_path (str): Path to save classification results
        log_file (str): Path to log file
        batch_size (int): Number of rows to process in each batch
        resume (bool): Whether to resume from previous run
    
    Returns:
        DataFrame with classification results
    """
    # Start timing
    start_time = time.time()
    
    # Prepare logging
    log_mode = 'a' if resume else 'w'
    with open(log_file, log_mode, encoding='utf-8') as log:
        if not resume:
            log.write(f"Full Dataset Validation Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Dataset: {dataset_path}\n\n")
    
    try:
        # Read the dataset
        df = pd.read_csv(dataset_path)
        total_rows = len(df)
        
        # Determine start index for resuming
        start_index = 0
        if resume and os.path.exists(output_path):
            existing_results = pd.read_csv(output_path)
            start_index = len(existing_results)
            log_mode = 'a'
        
        # Prepare classification components
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
        
        # Prepare results storage
        results = []
        classification_times = []
        
        # Process dataset in batches
        for batch_start in range(start_index, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\nProcessing Batch: {batch_start} to {batch_end}\n")
            
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Time individual classification
                    classification_start = time.time()
                    result = classification_agent.initial_classification({'input': row['post']})
                    classification_end = time.time()
                    
                    # Log classification details
                    classification_time = classification_end - classification_start
                    classification_times.append(classification_time)
                    
                    batch_results.append({
                        'id': row['id'],
                        'post': row['post'],
                        'category': result['final_classification']
                    })
                    
                except Exception as row_error:
                    # Log individual row errors
                    with open(log_file, 'a', encoding='utf-8') as log:
                        log.write(f"Error classifying row {idx}: {row_error}\n")
                        log.write(f"Traceback: {traceback.format_exc()}\n")
                    
                    # Add a default classification in case of error
                    batch_results.append({
                        'id': row['id'],
                        'post': row['post'],
                        'category': 0  # Default to irrelevant
                    })
            
            # Append batch results
            results.extend(batch_results)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            
            # Log progress
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"Processed {batch_end} / {total_rows} rows\n")
                log.write(f"Current Results Shape: {results_df.shape}\n")
        
        # Final logging
        with open(log_file, 'a', encoding='utf-8') as log:
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze classification distribution
            class_distribution = results_df['category'].value_counts(normalize=True)
            
            log.write("\nFinal Performance Summary:\n")
            log.write(f"Total Validation Time: {total_time:.2f} seconds\n")
            log.write(f"Average Classification Time: {sum(classification_times)/len(classification_times):.4f} seconds\n")
            log.write("Classification Distribution:\n")
            for label, proportion in class_distribution.items():
                log.write(f"  Label {label}: {proportion*100:.2f}%\n")
            
            log.write(f"\nFinal results saved to: {output_path}\n")
        
        print(f"Full dataset validation complete. Check {log_file} for details.")
        return results_df
    
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"Critical Error during validation: {e}\n")
            log.write(f"Traceback: {traceback.format_exc()}\n")
        raise

# If script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate full dataset')
    parser.add_argument('--dataset', default='/home/dukhanin/soc_hack/docs/Хакатон_2024/dataset_select_hackathon_26_11.csv', help='Path to dataset')
    parser.add_argument('--output', default='classification_results.csv', help='Output CSV path')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    
    args = parser.parse_args()
    
    validate_full_dataset(
        dataset_path=args.dataset, 
        output_path=args.output, 
        resume=args.resume,
        batch_size=args.batch_size
    ) 