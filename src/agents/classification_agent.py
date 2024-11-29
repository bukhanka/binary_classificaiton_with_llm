from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional, Dict, Any
import pandas as pd
import yaml
import os

from src.utils.vector_store import VectorStoreRetriever

class AgentState(TypedDict):
    input: str
    intermediate_steps: List[str]
    final_classification: Optional[int]
    confidence_score: Optional[float]
    retrieved_examples: Optional[List[Dict[str, Any]]]
    label_distribution: Optional[Dict[int, float]]

class ChainOfThoughtPrompt:
    @staticmethod
    def generate_reasoning_prompt(
        text: str, 
        similar_examples: List[Dict[str, Any]], 
        label_distribution: Dict[int, float]
    ) -> str:
        """
        Generate a comprehensive reasoning prompt for GPT-4o
        
        Args:
            text (str): Input text to classify
            similar_examples (List[Dict]): Retrieved similar examples
            label_distribution (Dict): Distribution of labels in similar examples
        
        Returns:
            Detailed reasoning prompt
        """
        # Prepare similar examples text
        examples_text = "\n".join([
            f"Example (Label: {ex['label']}, Similarity: {ex.get('distance', 'N/A')}):\n{ex['text']}"
            for ex in similar_examples
        ])
        
        # Prepare label distribution text
        label_dist_text = "\n".join([
            f"Label {label}: {prob * 100:.2f}%" 
            for label, prob in label_distribution.items()
        ])
        
        return f"""You are an expert text classifier specialized in student communication analysis. 
Your task is to perform a nuanced, multi-dimensional classification of student social media posts.

CLASSIFICATION GUIDELINES:
1. Relevance Criteria:
   - Relevant posts contain:
     a) Student opinions, experiences, or feedback
     b) Discussions about university life
     c) Genuine concerns or observations
     d) Personal reflections on educational experiences

   - Irrelevant posts include:
     a) Advertisements
     b) Pure informational announcements
     c) Generic questions without context
     d) Spam or unrelated content
     e) Purely social interactions without educational context

2. Contextual Analysis Framework:
   - Semantic Depth: Analyze beyond surface-level text
   - Thematic Resonance: Identify underlying educational themes
   - Contextual Nuance: Consider implicit and explicit meanings

INPUT TEXT:
{text}

CONTEXTUAL EVIDENCE:
- Similar Examples:
{examples_text}

- Label Distribution of Similar Examples:
{label_dist_text}

REASONING PROTOCOL:
1. Semantic Decomposition
   - Break down text into core semantic units
   - Identify potential educational relevance markers
   - Assess communicative intent

2. Comparative Analysis
   - Compare with retrieved similar examples
   - Evaluate thematic alignment
   - Detect subtle contextual indicators

3. Multi-Dimensional Scoring
   - Educational Relevance Score (0-1)
   - Thematic Coherence Assessment
   - Contextual Significance Evaluation

4. Confidence Calibration
   - Synthesize multi-level insights
   - Provide transparent reasoning
   - Quantify classification confidence

RESPONSE FORMAT (Strict JSON):
{{
    "label": 0 or 1,  // 1: Relevant, 0: Irrelevant
    "confidence": 0.00-1.00,  // Confidence score
    "reasoning": "Comprehensive explanation of classification decision",
    "key_factors": [
        "Factor 1 description",
        "Factor 2 description"
    ]
}}

CRITICAL INSTRUCTIONS:
- Be precise and analytical
- Avoid ambiguity
- Provide clear, justifiable reasoning
- Consider the broader context of student communication
"""

class ClassificationAgent:
    def __init__(
        self, 
        feature_extractor, 
        classifier, 
        relevant_df: pd.DataFrame, 
        irrelevant_df: pd.DataFrame,
        config_path: str = 'config.yaml',
        validation_dataset: pd.DataFrame = None
    ):
        """
        Initialize classification agent with configuration
        
        Args:
            feature_extractor: Feature extraction utility
            classifier: Machine learning classifier
            relevant_df: DataFrame of relevant examples
            irrelevant_df: DataFrame of irrelevant examples
            config_path (str): Path to configuration file
            validation_dataset (pd.DataFrame, optional): Validation dataset
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Use GPT-4o-mini with configuration settings
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',  # Specify GPT-4o-mini
            temperature=self.config['system']['llm_temperature'],
            max_tokens=512
        )
        
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        
        # Initialize Vector Store
        self.vector_store = VectorStoreRetriever(
            config_path=config_path,
            validation_dataset=validation_dataset
        )
        self.vector_store.create_collection()
        
        # Populate Vector Store
        self._populate_vector_store(relevant_df, irrelevant_df)
    
    def _populate_vector_store(
        self, 
        relevant_df: pd.DataFrame, 
        irrelevant_df: pd.DataFrame
    ):
        """
        Populate vector store with training examples
        
        Args:
            relevant_df (pd.DataFrame): Relevant examples
            irrelevant_df (pd.DataFrame): Irrelevant examples
        """
        # If no dataframes provided, use validation dataset
        if relevant_df.empty and irrelevant_df.empty and self.vector_store.validation_dataset is not None:
            combined_df = self.vector_store.validation_dataset
        else:
            # Combine datasets
            combined_df = pd.concat([relevant_df, irrelevant_df])
        
        # Add documents to vector store
        self.vector_store.add_documents(
            documents=combined_df['post'].tolist(),
            labels=combined_df['label'].tolist() if 'label' in combined_df.columns else [0] * len(combined_df)
        )
    
    def initial_classification(self, state: AgentState):
        """
        Perform classification using vector store retrieval and GPT-4o reasoning
        
        Args:
            state (AgentState): Current agent state
        
        Returns:
            Updated agent state
        """
        text = state['input']
        
        # Retrieve Similar Examples
        retrieval_results = self.vector_store.query(text, n_results=5)
        
        # Get Label Distribution
        label_distribution = self.vector_store.get_label_distribution(retrieval_results)
        
        # Prepare Similar Examples
        similar_examples = [
            {
                'text': doc, 
                'label': metadata['label'], 
                'distance': distance
            }
            for doc, metadata, distance in zip(
                retrieval_results['documents'], 
                retrieval_results['metadatas'], 
                retrieval_results['distances']
            )
        ]
        
        # Generate Reasoning Prompt
        reasoning_prompt = ChainOfThoughtPrompt.generate_reasoning_prompt(
            text, similar_examples, label_distribution
        )
        
        # LLM Reasoning with JSON output
        llm_response = self.llm.invoke(reasoning_prompt)
        
        # Parse JSON response
        try:
            classification_result = eval(llm_response.content)
            final_label = classification_result['label']
            confidence = classification_result['confidence']
            reasoning = classification_result['reasoning']
        except Exception as e:
            # Fallback to default classification
            final_label = self.config['classification']['default_label']
            confidence = 0.5
            reasoning = "Error in parsing LLM response"
        
        # Apply confidence threshold
        if confidence < self.config['classification']['confidence_threshold']:
            final_label = self.config['classification']['default_label']
        
        return {
            **state,
            'intermediate_steps': [
                f"Retrieved Examples: {similar_examples}",
                f"Label Distribution: {label_distribution}",
                f"LLM Reasoning: {reasoning}"
            ],
            'retrieved_examples': similar_examples,
            'label_distribution': label_distribution,
            'final_classification': final_label,
            'confidence_score': confidence
        }
    
    def create_graph(self):
        """
        Create LangGraph workflow for classification
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("initial_classification", self.initial_classification)
        workflow.set_entry_point("initial_classification")
        workflow.add_edge("initial_classification", END)
        
        return workflow.compile() 