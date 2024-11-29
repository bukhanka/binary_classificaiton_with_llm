import os
import pandas as pd
import yaml

class DatasetPreparation:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def create_validation_dataset(
        self, 
        input_dataset_path: str = None, 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        """
        Create a validation dataset from the input dataset
        
        Args:
            input_dataset_path (str): Path to input dataset
            test_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
        """
        # Use path from config if not provided
        if input_dataset_path is None:
            input_dataset_path = self.config['datasets']['input_dataset']
        
        # Read input dataset
        combined_df = pd.read_csv(input_dataset_path)
        
        # Ensure required columns
        if 'id' not in combined_df.columns:
            combined_df['id'] = range(1, len(combined_df) + 1)
        
        if 'post' not in combined_df.columns:
            raise ValueError("Input dataset must have a 'post' column")
        
        # Shuffle dataset
        combined_df = combined_df.sample(frac=1, random_state=random_state)
        
        # Save validation dataset
        validation_path = self.config['datasets']['validation_dataset']
        combined_df.to_csv(validation_path, index=False, encoding='utf-8')
        
        print(f"Validation dataset created: {validation_path}")
        print(f"Total samples: {len(combined_df)}")
        
        return combined_df