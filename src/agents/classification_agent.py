from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Optional, Dict, Any
import pandas as pd
import yaml
import os
import logging
import ast
import time

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
        Generate a comprehensive reasoning prompt for GPT-4o with specific class descriptions
        
        Args:
            text (str): Input text to classify
            similar_examples (List[Dict]): Retrieved similar examples
            label_distribution (Dict): Distribution of labels in similar examples
        
        Returns:
            Detailed reasoning prompt
        """
        # Detailed class descriptions and examples
        class_descriptions = {
            0: {
                "name": "Учебная и внеучебная вовлеченность",
                "description": """Отношение к преподавателям и качеству преподавания в университете, вклю��ая:
                1. Конкретные отзывы о качестве преподавания и преподавателях
                2. Системные проблемы образовательного процесса (сессия, задания, экзамены)
                3. Технические аспекты обучения (документооборот, личные кабинеты)
                4. Внеучебная деятельность (профкомы, кружки, волонтерство)
                5. Студенческие мероприятия и активности""",
                "positive_examples": [
                    "Хочу сказать большое спасибо кафедре взрослых инфекционных болезней! Дорогие преподаватели, Вы хорошо обучаете во время всего цикла, отличные презентации!",
                    "У Натальи Михайловны очень интересные лекции, с живой подачей материала...приходите",
                    "«языковой империализм» английского языка поставил на мне своё клеймо, а всё потому, что у меня не было права отказаться от этого предмета"
                ],
                "negative_examples": [
                    "Владислав Иванов лучший преподаватель! Группа 5116 так считает! Ps: Петров Никита",
                    "⚡Иванова Даша⚡ САМЫЙ ЛУЧШИЙ ПРОФОРГ НА СВЕТЕ",
                    "Кто-нибудь сдавал экзамен по бухгалтерскому учету у Романенко О.Е.? Сложно?"
                ]
            },
            1: {
                "name": "Социально-бытовые условия",
                "description": """Условия проживания и быта студентов, включая:
                1. Проблемы общежития (ремонт, оборудование, условия)
                2. Доступность и качество кампуса
                3. Организация питания
                4. Медицинское обслуживание
                5. Инфраструктура университета""",
                "positive_examples": [
                    "С 1 сентября! Когда включат отопле��ие в общаге? Мы умираем от холода 🥶",
                    "В общежитии опять проблемы с горячей водой, когда это закончится?",
                    "Почему в столовой такие большие очереди? Невозможно успеть поесть между парами"
                ],
                "negative_examples": [
                    "Кто хочет пойти на концерт Schokk'a 17 февраля?",
                    "Принимает ли Семенов А.М. автоматы от других преподавателей?",
                    "В класс Марии Владимировны требуется концертмейстер"
                ]
            },
            2: {
                "name": "Финансовые условия",
                "description": """Финансовые аспекты обучения и студенческой жизни:
                1. Стипендии и социальные выплаты
                2. Материальная помощь
                3. Оплата общежития
                4. Стоимость обучения
                5. Возможности подработки""",
                "positive_examples": [
                    "Почему сиротам пришла только часть стипендии, а именно только социальная. На сайте другая сумма.",
                    "Когда будет выплата материальной помощи? Уже третий месяц жду",
                    "Подскажите, как можно получить социальную стипендию?"
                ],
                "negative_examples": [
                    "Кто хочет пойти на концерт в субботу?",
                    "Где можно найти расписание на следующую неделю?",
                    "Кто знает телефон деканата?"
                ]
            },
            3: {
                "name": "Лояльность к ВУЗу",
                "description": """Общее отношение к университету и его репутации:
                1. Эмоциональная привязанность к вузу
                2. Оценка общего уровня университета
                3. Отношение к руководству вуза
                4. Гордость за достижения университета
                5. Критика общего состояния вуза
                6. Сравнение с другими вузами""",
                "positive_examples": [
                    "Горжусь родным вузом! 🫶",
                    "У меня слёзы, мурашки и гордость за самый лучший ВУЗ❤ Спасибо, Императорский!",
                    "Как приятно, что наша академия расширяет границы своего влияния"
                ],
                "negative_examples": [
                    "отчисляйся, универ - днище полное, только выйграешь",
                    "Ничего не работает и не решается. Этот вуз уже не спасти.",
                    "пол студентов в эту шарагу поступают и не знают для чего."
                ]
            }
        }

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
        
        # Prepare class descriptions text
        class_desc_text = "\n\n".join([
            f"CLASS {label}: {desc['name']}\n"
            f"Description: {desc['description']}\n"
            f"Positive Examples: {desc['positive_examples']}\n"
            f"Negative Examples: {desc['negative_examples']}"
            for label, desc in class_descriptions.items()
        ])

        return f"""You are an expert text classifier specialized in student communication analysis. 
Your task is to perform a nuanced, multi-dimensional classification of student social media posts.

CLASSIFICATION CLASSES:
{class_desc_text}

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

INPUT TEXT:
{text}

CONTEXTUAL EVIDENCE:
- Similar Examples:
{examples_text}

- Label Distribution of Similar Examples:
{label_dist_text}

REASONING PROTOCOL:
1. Carefully analyze the input text against the class descriptions
2. Identify the most appropriate class based on thematic alignment
3. Provide a clear justification for your classification

RESPONSE FORMAT (Strict JSON):
{{
    "label": 0 or 1 or 2 or 3,  // Corresponding class label
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
            max_tokens=512,
            api_key=os.getenv('OPENAI_API_KEY')  # Ensure API key is set
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
    
    def initial_classification(self, state: AgentState, max_retries=3):
        """
        Perform classification using vector store retrieval and GPT-4o reasoning
        
        Args:
            state (AgentState): Current agent state
            max_retries (int): Number of retry attempts
        
        Returns:
            Updated agent state
        """
        for attempt in range(max_retries):
            try:
                # Existing classification logic remains the same
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
                
                # Robust parsing with multiple fallback methods
                try:
                    # First, try ast.literal_eval
                    classification_result = ast.literal_eval(llm_response.content)
                    
                    # Validate required keys
                    required_keys = ['label', 'confidence', 'reasoning']
                    for key in required_keys:
                        if key not in classification_result:
                            raise KeyError(f"Missing required key: {key}")
                    
                    # If parsing and validation succeed, return the result
                    return {
                        **state,
                        'intermediate_steps': [
                            f"Retrieved Examples: {similar_examples}",
                            f"Label Distribution: {label_distribution}",
                            f"LLM Reasoning: {classification_result.get('reasoning', '')}"
                        ],
                        'retrieved_examples': similar_examples,
                        'label_distribution': label_distribution,
                        'final_classification': classification_result['label'],
                        'confidence_score': classification_result['confidence']
                    }
                
                except (SyntaxError, ValueError, KeyError) as e:
                    logging.warning(f"Parsing attempt {attempt + 1} failed: {e}")
                    
                    # Optional: Add a slight delay between retries
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
            
            except Exception as e:
                logging.error(f"Classification error on attempt {attempt + 1}: {e}")
        
        # If all retries fail, return default classification
        return {
            **state,
            'final_classification': self.config['classification']['default_label'],
            'confidence_score': 0.0,
            'error': "Failed to parse classification result after multiple attempts"
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