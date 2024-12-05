import os
import sys
import yaml
import pandas as pd
import numpy as np
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('hackathon_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class FastClassificationAgent:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize classification agent with efficient setup
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Predefined class descriptions
        self.class_descriptions = {
            0: "Учебная и внеучебная вовлеченность: отношение к преподавателям, качеству преподавания, системным проблемам образовательного процесса, внеучебной деятельности",
            1: "Социально-бытовые условия: проблемы общежития, кампуса, питания, медицинского обслуживания, инфраструктуры университета",
            2: "Финансовые условия: стипендии, социальные выплаты, материальная помощь, оплата общежития, возможности подработки",
            3: "Лояльность к ВУЗу: отношение к администрации, заинтересованность в развитии, уровень сопричастности, общая оценка университета"
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Efficiently classify text using async OpenAI call
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Lightweight, fast model
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system", 
                        "content": f"""
                        You are an expert text classifier for student communication. 
                        Classify the text into ONE of these categories:
                        0: {self.class_descriptions[0]}
                        1: {self.class_descriptions[1]}
                        2: {self.class_descriptions[2]}
                        3: {self.class_descriptions[3]}
                        
                        Provide a JSON response with:
                        - category: 0, 1, 2, or 3
                        - confidence: 0.0-1.0
                        - reasoning: brief explanation
                        """
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                max_tokens=256
            )
            
            # Parse response
            result = response.choices[0].message.content
            return self._parse_classification(result)
        
        except Exception as e:
            logging.error(f"Classification error: {e}")
            return {
                'category': -1,
                'confidence': 0.0,
                'reasoning': str(e)
            }
    
    def _parse_classification(self, result: str) -> Dict[str, Any]:
        """
        Parse classification result with robust error handling
        """
        try:
            import json
            parsed = json.loads(result)
            
            # Validate category
            category = int(parsed.get('category', -1))
            if category not in [0, 1, 2, 3]:
                category = -1
            
            return {
                'category': category,
                'confidence': float(parsed.get('confidence', 0.0)),
                'reasoning': parsed.get('reasoning', '')
            }
        
        except Exception as e:
            logging.warning(f"Parsing error: {e}")
            return {
                'category': -1,
                'confidence': 0.0,
                'reasoning': 'Parsing failed'
            }

async def process_batch(agent, batch: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process a batch of texts asynchronously
    """
    tasks = [agent.classify_text(text) for text in batch['post']]
    return await asyncio.gather(*tasks)

def generate_comprehensive_report(results_df: pd.DataFrame, start_time: float, end_time: float) -> str:
    """
    Generate a comprehensive report with detailed statistics
    """
    # Calculate processing time
    total_time = end_time - start_time
    
    # Category distribution
    category_counts = results_df['predicted_category'].value_counts()
    
    # Confidence statistics
    confidence_stats = results_df['confidence'].agg(['mean', 'median', 'min', 'max'])
    
    # Prepare report
    report = f"""
Classification Run Report
========================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Processing Statistics:
---------------------
Total Texts Processed: {len(results_df)}
Total Processing Time: {total_time:.2f} seconds
Average Processing Time per Text: {total_time / len(results_df):.4f} seconds

Category Distribution:
--------------------
{category_counts}

Confidence Metrics:
-----------------
Mean Confidence: {confidence_stats['mean']:.4f}
Median Confidence: {confidence_stats['median']:.4f}
Minimum Confidence: {confidence_stats['min']:.4f}
Maximum Confidence: {confidence_stats['max']:.4f}

Category Mapping:
---------------
0: Учебная и внеучебная вовлеченность
1: Социально-бытовые условия
2: Финансовые условия
3: Лояльность к ВУЗу

Notes:
------
- Predictions saved to hackathon_predictions.csv
- Detailed logs available in hackathon_prediction.log
"""
    return report

async def main():
    try:
        # Record start time
        start_time = time.time()
        
        # Load dataset
        dataset_path = '/home/dukhanin/soc_hack/data_2/Хакатон_2024_очный_этап/Хакатон_очный этап .csv'
        df = pd.read_csv(dataset_path, encoding='utf-8')
        
        # Initialize agent
        agent = FastClassificationAgent()
        
        # Batch processing
        batch_size = 50
        all_results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}")
            
            batch_results = await process_batch(agent, batch)
            all_results.extend(batch_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'id': df['id'],
            'post': df['post'],
            'predicted_category': [r['category'] for r in all_results],
            'confidence': [r['confidence'] for r in all_results],
            'reasoning': [r['reasoning'] for r in all_results]
        })
        
        # Record end time
        end_time = time.time()
        
        # Save results
        output_path = 'hackathon_predictions.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate and save comprehensive report
        report = generate_comprehensive_report(results_df, start_time, end_time)
        
        # Save report to text file
        report_path = 'classification_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Log to console and file
        logging.info(report)
        print(f"Predictions saved to {output_path}")
        print(f"Comprehensive report saved to {report_path}")
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main()) 