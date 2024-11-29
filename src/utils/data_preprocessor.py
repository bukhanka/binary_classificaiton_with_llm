import pandas as pd
import numpy as np
from typing import Tuple, List

class DataPreprocessor:
    @staticmethod
    def load_datasets(
        relevant_paths: List[str], 
        irrelevant_paths: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine relevant and irrelevant datasets
        
        Args:
            relevant_paths (List[str]): Paths to relevant post CSV files
            irrelevant_paths (List[str]): Paths to irrelevant post CSV files
        
        Returns:
            Tuple of DataFrames for relevant and irrelevant posts
        """
        relevant_dfs = [pd.read_csv(path) for path in relevant_paths]
        irrelevant_dfs = [pd.read_csv(path) for path in irrelevant_paths]
        
        relevant_df = pd.concat(relevant_dfs, ignore_index=True)
        irrelevant_df = pd.concat(irrelevant_dfs, ignore_index=True)
        
        relevant_df['label'] = 1
        irrelevant_df['label'] = 0
        
        return relevant_df, irrelevant_df
    
    @staticmethod
    def prepare_training_data(
        relevant_df: pd.DataFrame, 
        irrelevant_df: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation datasets
        
        Args:
            relevant_df (pd.DataFrame): Relevant posts DataFrame
            irrelevant_df (pd.DataFrame): Irrelevant posts DataFrame
            test_size (float): Proportion of data to use for validation
        
        Returns:
            Training and validation DataFrames for features and labels
        """
        combined_df = pd.concat([relevant_df, irrelevant_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42)  # Shuffle
        
        split_idx = int(len(combined_df) * (1 - test_size))
        
        train_df = combined_df.iloc[:split_idx]
        val_df = combined_df.iloc[split_idx:]
        
        return train_df, val_df 