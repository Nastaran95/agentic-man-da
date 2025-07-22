import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv(".env")

from app.agent import create_bot
from app.utils import generate_sqlite_table_info_query, format_table_info
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def load_table_info_from_data_dictionary(db_path: str) -> str:
    """
    Load table information from the data dictionary JSON file
    based on the database name
    """
    try:
        # Load the data dictionary
        with open("data/data_dictionary.json", "r") as f:
            data_dict = json.load(f)
        
        # Extract database name from path
        db_name = os.path.basename(db_path).replace(".db", ".")
        
        # Find the corresponding CSV file in the data dictionary
        csv_file = None
        for key in data_dict.keys():
            if db_name in key:
                csv_file = key
                break
        
        if csv_file and csv_file in data_dict:
            table_info = data_dict[csv_file]["table_info"]
            logger.info(f"Loaded table info from data dictionary for {csv_file}")
            return table_info
        else:
            logger.warning(f"No table info found in data dictionary for {db_name}, falling back to database generation")
            # Fallback to original method
            table_info_query = generate_sqlite_table_info_query(db_path)
            return format_table_info(table_info_query)
            
    except Exception as e:
        logger.error(f"Error loading table info from data dictionary: {e}")
        # Fallback to original method
        table_info_query = generate_sqlite_table_info_query(db_path)
        return format_table_info(table_info_query)

class ExcelQuestionProcessor:
    def __init__(self, excel_path: str, db_path: str, model: str):
        """
        Initialize the Excel question processor
        
        Args:
            excel_path: Path to the Excel file
            db_path: Path to the SQLite database
            model: Name of the LLM model to use
        """
        self.excel_path = excel_path
        self.db_path = db_path
        self.model = model
        self.bot = None
        self.results = []
        
    async def initialize_bot(self):
        """Initialize the chatbot with database schema"""
        try:
            # Load table info from data dictionary
            table_info = load_table_info_from_data_dictionary(self.db_path)
            print(f"Table info loaded: {table_info}")
            
            # Create the bot
            self.bot = await create_bot(table_info, self.db_path, self.model)
            logger.info("Bot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing bot: {e}")
            raise
    
    def read_excel_questions(self, tab_names: List[str]) -> List[Dict[str, Any]]:
        """
        Read questions from Excel file tabs
        
        Args:
            tab_names: List of tab names to read from
            
        Returns:
            List of dictionaries containing question data
        """
        questions = []
        
        try:
            # Read each tab
            for tab_name in tab_names:
                logger.info(f"Reading tab: {tab_name}")
                df = pd.read_excel(self.excel_path, sheet_name=tab_name)
                
                # Process each row
                for index, row in df.iterrows():
                    question_data = {
                        'question_id': f"{tab_name}_{index + 1}",
                        'question': str(row.get('Question', '')),
                        'answer_correct': str(row.get('Answer', '')),
                        'tab_name': tab_name,
                        'row_index': index + 1
                    }
                    questions.append(question_data)
                    
            logger.info(f"Read {len(questions)} questions from {len(tab_names)} tabs")
            return questions
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    async def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single question through the agent
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with results including timing and cost
        """
        
        
        try:
            # Send question to agent
            question = question_data['question']
            logger.info(f"Processing question {question_data['question_id']}: {question[:50]}...")
            
            start_time = time.time()

            # Get response from bot
            self.bot.clear_conversation_history()
            response_message, steps = await self.bot(question)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract agent's final answer (last AI message)
            agent_answer = ""
            if response_message and hasattr(response_message, 'content'):
                agent_answer = response_message.content
            
            # Fetch cost and token usage from LangSmith
            cost, tokens = self.get_latest_langsmith_cost_and_tokens()
            
            result = {
                'question_id': question_data['question_id'],
                'question': question_data['question'],
                'answer_correct': question_data['answer_correct'],
                'agent_final_answer': agent_answer,
                'time_seconds': round(processing_time, 2),
                'cost_estimate': cost,
                'num_tokens': tokens,
                'tab_name': question_data['tab_name'],
                'row_index': question_data['row_index'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Completed question {question_data['question_id']} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question {question_data['question_id']}: {e}")
            return {
                'question_id': question_data['question_id'],
                'question': question_data['question'],
                'answer_correct': question_data['answer_correct'],
                'agent_final_answer': f"ERROR: {str(e)}",
                'time_seconds': round(time.time() - start_time, 2),
                'cost_estimate': 0.0,
                'num_tokens': 0,
                'tab_name': question_data['tab_name'],
                'row_index': question_data['row_index'],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_latest_langsmith_cost_and_tokens(self):
        """
        Fetch the latest run from LangSmith and return cost and token usage.
        """
        try:
            project_name = os.environ.get("LANGSMITH_PROJECT")
            if not project_name:
                logger.warning("LANGSMITH_PROJECT environment variable not set.")
                return None, None
            client = Client()
            time.sleep(3)
            runs = client.list_runs(project_name=project_name, limit=1 , is_root=1)
            run = next(runs, None)
            if run:
                return run.total_cost, run.total_tokens
            else:
                logger.warning("No LangSmith run found.")
                return None, None
        except Exception as e:
            logger.error(f"Error fetching LangSmith run: {e}")
            return None, None
    
    
    
    async def process_all_questions(self, tab_names: List[str], output_file: str = None):
        """
        Process all questions from the Excel file
        
        Args:
            tab_names: List of tab names to process
            output_file: Path to save results (optional)
        """
        if not self.bot:
            await self.initialize_bot()
        
        # Read questions from Excel
        questions = self.read_excel_questions(tab_names)
        
        logger.info(f"Starting to process {len(questions)} questions...")
        
        # Process each question
        for i, question_data in enumerate(questions):
            result = await self.process_question(question_data)
            self.results.append(result)
            
            # Log progress
            if (i + 1) % BATCH_SIZE == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")
            
            # if i>2:
            #     break
        
        # Save results
        if output_file:
            self.save_results(output_file)
        
        logger.info(f"Completed processing {len(questions)} questions")
        return self.results
    
    def save_results(self, output_file: str):
        """
        Save results to a file
        
        Args:
            output_file: Path to save the results
        """
        try:
            # Create DataFrame from results
            df = pd.DataFrame(self.results)
            
            # Save to Excel
            if output_file.endswith('.xlsx'):
                df.to_excel(output_file, index=False)
            elif output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
            else:
                # Default to Excel
                output_file = output_file + '.xlsx'
                df.to_excel(output_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
            # Print summary
            total_time = sum(r['time_seconds'] for r in self.results)
            # total_cost = sum(r['cost_estimate'] for r in self.results)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            # logger.info(f"Total estimated cost: ${total_cost:.6f}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

async def main():
    """
    Main function to run the Excel question processor
    """
    # Create processor using config
    processor = ExcelQuestionProcessor(EXCEL_PATH, DB_PATH, MODEL_NAME)
    
    try:
        # Process all questions
        results = await processor.process_all_questions(TAB_NAMES, OUTPUT_FILE)
        
        print(f"\nProcessing completed!")
        print(f"Total questions processed: {len(results)}")
        print(f"Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 