import os
import sys
import yaml
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score, classification_report
from src.agents.classification_agent import ClassificationAgent, AgentState
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
    
    Returns:
        dict: Loaded configuration
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def configure_openai_api():
    """
    Configure OpenAI API key from .env file
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load .env file
    except ImportError:
        print("python-dotenv not installed. Please install it with 'pip install python-dotenv'")
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("""
        No OpenAI API key found in .env file.
        Please ensure your .env file contains:
        OPENAI_API_KEY=your_api_key_here
        """)
    
    # Optional: Trim any whitespace or 'Bearer ' prefix
    api_key = api_key.strip().replace('Bearer ', '')
    os.environ['OPENAI_API_KEY'] = api_key
    return api_key

def load_test_datasets(config):
    """
    Load test datasets from configuration
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        pd.DataFrame: Combined test dataset with labels
    """
    dataset_paths = config['datasets']['dataset_paths']
    
    test_data = []
    for idx, filepath in enumerate(dataset_paths):
        df = pd.read_csv(filepath)
        df['label'] = idx  # Assign label based on file order
        df['class_name'] = config['classification']['classes'][idx]
        test_data.append(df)
    
    return pd.concat(test_data, ignore_index=True)

def classify_single_text(agent, text, config):
    """
    Classify a single text with error handling
    
    Args:
        agent (ClassificationAgent): Classification agent
        text (str): Text to classify
        config (dict): Configuration dictionary
    
    Returns:
        dict: Classification result
    """
    # Prepare agent state
    state = {
        'input': text,
        'intermediate_steps': [],
        'final_classification': None,
        'confidence_score': None,
        'retrieved_examples': None,
        'label_distribution': None
    }
    
    # Perform classification with error handling
    try:
        result = agent.initial_classification(state)
        
        # Extract predicted label and confidence
        predicted_label = result.get('final_classification', -1)
        confidence = result.get('confidence_score', 0.0)
        
        # Map label to class name
        predicted_class_name = config['classification']['classes'].get(predicted_label, 'Unknown')
        
        return {
            'predicted_label': predicted_label,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'success': True
        }
    
    except Exception as e:
        print(f"Classification error for text: {text}")
        print(f"Error: {e}")
        return {
            'predicted_label': -1,
            'predicted_class_name': 'Unknown',
            'confidence': 0.0,
            'success': False,
            'error': str(e)
        }

def validate_classification(agent, test_dataset, config, max_workers=None):
    """
    Validate classification agent performance using parallel processing
    
    Args:
        agent (ClassificationAgent): Trained classification agent
        test_dataset (pd.DataFrame): Test dataset to validate
        config (dict): Configuration dictionary
        max_workers (int, optional): Maximum number of worker threads
    
    Returns:
        dict: Validation metrics
    """
    # Use number of CPU cores if max_workers not specified
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Prepare data for parallel processing
    texts = test_dataset['post'].tolist()
    true_labels = test_dataset['label'].tolist()
    true_class_names = test_dataset['class_name'].tolist()
    
    # Parallel classification
    predicted_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit classification tasks
        future_to_text = {
            executor.submit(classify_single_text, agent, text, config): text 
            for text in texts
        }
        
        # Collect results
        for future in as_completed(future_to_text):
            result = future.result()
            predicted_results.append(result)
    
    # Process results
    predicted_labels = []
    predicted_class_names = []
    confidences = []
    
    for result in predicted_results:
        if result['success']:
            predicted_labels.append(result['predicted_label'])
            predicted_class_names.append(result['predicted_class_name'])
            confidences.append(result['confidence'])
    
    # Filter out any failed classifications
    valid_indices = [i for i, label in enumerate(true_labels) if 
                     predicted_labels[i] != -1]
    
    true_labels = [true_labels[i] for i in valid_indices]
    true_class_names = [true_class_names[i] for i in valid_indices]
    predicted_labels = [predicted_labels[i] for i in valid_indices]
    predicted_class_names = [predicted_class_names[i] for i in valid_indices]
    
    # Calculate metrics
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=list(config['classification']['classes'].values())
    )
    
    return {
        'f1_score': f1,
        'classification_report': report,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'predicted_class_names': predicted_class_names,
        'true_class_names': true_class_names,
        'average_confidence': np.mean(confidences) if confidences else 0.0
    }

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Configure OpenAI API Key
        configure_openai_api()
        
        # Load test datasets
        test_dataset = load_test_datasets(config)
        
        # Initialize classification agent
        agent = ClassificationAgent(
            feature_extractor=None,  # Replace with your feature extractor
            classifier=None,  # Replace with your classifier
            relevant_df=pd.DataFrame(),  # Provide training data if available
            irrelevant_df=pd.DataFrame(),
            config_path='config.yaml',
            validation_dataset=test_dataset
        )
        
        # Validate classification with parallel processing
        validation_results = validate_classification(
            agent, 
            test_dataset, 
            config, 
            max_workers=os.cpu_count() or 4  # Use all available CPU cores
        )
        
        # Print results
        print("Validation Results:")
        print(f"F1 Score (Macro): {validation_results['f1_score']}")
        print("\nClassification Report:")
        print(validation_results['classification_report'])
        print(f"\nAverage Confidence: {validation_results['average_confidence']}")
        
        # Optional: Save results to file
        with open('validation_results.txt', 'w') as f:
            f.write(f"F1 Score (Macro): {validation_results['f1_score']}\n")
            f.write("\nClassification Report:\n")
            f.write(validation_results['classification_report'])
            f.write(f"\nAverage Confidence: {validation_results['average_confidence']}")

    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 