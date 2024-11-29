import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings

class FeatureExtractor:
    def __init__(self, model_name: str = 'text-embedding-3-large'):
        """
        Initialize feature extractor with OpenAI embeddings
        
        Args:
            model_name (str): OpenAI embedding model name
        """
        self.embedding_model = OpenAIEmbeddings(model=model_name)
    
    def extract_features(self, texts: pd.Series) -> np.ndarray:
        """
        Extract embeddings for given texts
        
        Args:
            texts (pd.Series): Series of text data
        
        Returns:
            numpy array of embeddings
        """
        # Convert texts to list and generate embeddings
        embeddings = self.embedding_model.embed_documents(texts.tolist())
        return np.array(embeddings)
    
    def prepare_features(self, df: pd.DataFrame, text_column: str) -> dict:
        """
        Prepare features and labels for model training
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Column containing text data
        
        Returns:
            Dictionary with features and labels
        """
        features = self.extract_features(df[text_column])
        labels = df['label'].values
        
        return {
            'features': features,
            'labels': labels
        } 