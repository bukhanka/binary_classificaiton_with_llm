import os
import hashlib
import pickle
import chromadb
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
import time
import logging

class VectorStoreRetriever:
    def __init__(
        self, 
        config_path: str = 'config.yaml',
        validation_dataset: pd.DataFrame = None
    ):
        """
        Initialize vector store retriever with configuration
        
        Args:
            config_path (str): Path to configuration file
            validation_dataset (pd.DataFrame, optional): Validation dataset
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Embedding cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'embedding_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if semantic search is enabled
        self.use_semantic_search = self.config['system']['use_semantic_search']
        
        # OpenAI Embedding Model
        self.embedding_model = OpenAIEmbeddings(
            model=self.config['system']['embedding_model']
        )
        
        # Chroma client with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        
        # Collection for storing documents
        self.collection = None
        
        # Validation dataset
        self.validation_dataset = validation_dataset
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_embedding_cache_path(self, text: str) -> str:
        """
        Generate a unique cache path for an embedding
        
        Args:
            text (str): Text to generate cache path for
        
        Returns:
            Path to cache file
        """
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}_embedding.pkl")
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """
        Cache an embedding to disk
        
        Args:
            text (str): Original text
            embedding (List[float]): Embedding vector
        """
        cache_path = self._get_embedding_cache_path(text)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    
    def _load_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Load a cached embedding
        
        Args:
            text (str): Original text
        
        Returns:
            Cached embedding or None if not found
        """
        cache_path = self._get_embedding_cache_path(text)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def create_collection(
        self, 
        name: str = 'classification_examples', 
        metadata: Dict[str, str] = None
    ):
        """
        Create or get a Chroma collection
        
        Args:
            name (str): Name of the collection
            metadata (dict): Optional metadata for the collection
        """
        # Always provide default metadata if none is given
        if metadata is None:
            metadata = {
                "description": "Classification examples collection",
                "created_at": str(datetime.now()),
                "type": "text_classification"
            }
        
        try:
            # Try to get existing collection, create if not exists
            self.collection = self.chroma_client.get_or_create_collection(
                name=name, 
                metadata=metadata
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            # Fallback to creating a new collection with a unique name
            unique_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.collection = self.chroma_client.create_collection(
                name=unique_name, 
                metadata=metadata
            )
    
    def _safe_embed_documents(
        self, 
        documents: List[str], 
        max_retries: int = 3, 
        backoff_factor: float = 2.0
    ) -> List[List[float]]:
        """
        Safely generate embeddings with retry mechanism
        
        Args:
            documents (List[str]): Documents to embed
            max_retries (int): Maximum number of retries
            backoff_factor (float): Exponential backoff factor
        
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for doc in documents:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Check cached embedding first
                    cached_embedding = self._load_cached_embedding(doc)
                    if cached_embedding:
                        embeddings.append(cached_embedding)
                        break
                    
                    # Generate new embedding
                    new_embedding = self.embedding_model.embed_documents([doc])[0]
                    self._cache_embedding(doc, new_embedding)
                    embeddings.append(new_embedding)
                    break
                
                except (RateLimitError, APIConnectionError) as api_error:
                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    self.logger.warning(f"Embedding error (attempt {retry_count}): {api_error}")
                    self.logger.info(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                
                except Exception as e:
                    self.logger.error(f"Unexpected error embedding document: {e}")
                    # Add a zero vector as a fallback
                    embeddings.append([0.0] * 1536)  # Assuming 1536-dimensional embedding
                    break
            
            else:
                self.logger.error(f"Failed to embed document after {max_retries} attempts")
                # Add a zero vector as a final fallback
                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def add_documents(
        self, 
        documents: List[str], 
        labels: List[int], 
        metadata: List[Dict[str, Any]] = None,
        max_documents: int = 100  # Limit number of documents to embed
    ):
        """
        Add documents to the vector store with robust embedding
        
        Args:
            documents (List[str]): List of text documents
            labels (List[int]): Corresponding labels
            metadata (List[Dict]): Optional metadata for each document
            max_documents (int): Maximum number of documents to embed
        """
        # Skip if semantic search is disabled or no documents
        if not self.use_semantic_search or not documents:
            return
        
        if not self.collection:
            # Check if collection already exists
            try:
                self.collection = self.chroma_client.get_collection('validation_collection')
                return  # Collection already exists, no need to re-add
            except:
                self.collection = self.chroma_client.create_collection(name='validation_collection')
        
        # Ensure documents are not empty and unique
        unique_docs = []
        unique_labels = []
        unique_metadata = []
        seen_docs = set()
        
        for doc, label, meta in zip(documents, labels, metadata or [{'label': label} for label in labels]):
            # Clean and validate document
            if not doc or not isinstance(doc, str):
                continue
            
            # Remove leading/trailing whitespace and convert to lowercase
            cleaned_doc = doc.strip().lower()
            
            # Skip if document is already seen
            if cleaned_doc in seen_docs:
                continue
            
            unique_docs.append(doc)
            unique_labels.append(label)
            unique_metadata.append(meta)
            seen_docs.add(cleaned_doc)
            
            # Stop if we've reached max documents
            if len(unique_docs) >= max_documents:
                break
        
        # Skip if no valid documents
        if not unique_docs:
            return
        
        # Generate embeddings using safe method
        try:
            embeddings = self._safe_embed_documents(unique_docs)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return
        
        # Generate unique IDs using hash of document content
        ids = [
            f'doc_{hashlib.md5(doc.encode()).hexdigest()[:10]}'
            for doc in unique_docs[:len(embeddings)]
        ]
        
        # Add to Chroma
        self.collection.add(
            embeddings=embeddings,
            documents=unique_docs[:len(embeddings)],
            ids=ids,
            metadatas=unique_metadata[:len(embeddings)]
        )
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5, 
        filter_label: int = None
    ) -> Dict[str, Any]:
        """
        Retrieve most similar documents
        
        Args:
            query_text (str): Text to find similar documents for
            n_results (int): Number of results to return
            filter_label (int, optional): Filter results by specific label
        
        Returns:
            Dictionary with retrieved documents and their details
        """
        # If semantic search is disabled, return empty results
        if not self.use_semantic_search:
            return {
                'documents': [],
                'distances': [],
                'metadatas': []
            }
        
        # If validation dataset is available, use 'post' column
        if self.validation_dataset is not None:
            documents = self.validation_dataset['post'].tolist()
            labels = self.validation_dataset['label'].tolist() if 'label' in self.validation_dataset.columns else [0] * len(self.validation_dataset)
            
            # Regenerate collection with validation dataset
            self.collection = None
            self.add_documents(documents, labels)
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query_text)
        
        # Prepare query parameters
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': min(n_results, self.config['system']['max_similar_examples'])
        }
        
        # Add label filter if specified
        if filter_label is not None:
            query_params['where'] = {'label': filter_label}
        
        # Perform query
        results = self.collection.query(**query_params)
        
        # Process and return results
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }
    
    def get_label_distribution(self, results: Dict[str, Any]) -> Dict[int, float]:
        """
        Analyze label distribution of retrieved documents
        
        Args:
            results (Dict): Results from query method
        
        Returns:
            Dictionary of label probabilities
        """
        # If no results or semantic search disabled
        if not results['metadatas'] or not self.use_semantic_search:
            return {0: 0.5, 1: 0.5}
        
        labels = [metadata['label'] for metadata in results['metadatas']]
        unique_labels = set(labels)
        
        label_dist = {
            label: labels.count(label) / len(labels) 
            for label in unique_labels
        }
        
        return label_dist