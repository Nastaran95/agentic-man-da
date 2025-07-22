import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langsmith import Client
import openpyxl
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv(".env")
from app.tools import run_sqlite_query  # Add this import
import re
import asyncio
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QuestionData:
    question_id: str
    question: str
    answer_correct: str
    tab_name: str
    row_index: int

class DirectCodeLLMProcessor:
    """
    A processor that uses direct code approach to solve questions from Excel/CSV files
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name


        if "gemini" in model_name:
            print("------------------------------------------- gemini")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=0,
            )
        elif "llama" in model_name:
            print("------------------------------------------- llama")
            self.llm = ChatTogether(
                model=model_name,
                together_api_key=os.environ.get("TOGETHER_API_KEY"),
                temperature=0,
            )
        else:
            print("------------------------------------------- gpt")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0
            )

        
        self.langsmith_client = None
        
        # Initialize LangSmith client if available
        try:
            self.langsmith_client = Client()
        except Exception as e:
            logger.warning(f"LangSmith client initialization failed: {e}")
    
   
    def read_excel_questions(self, question_path: str, tab_names: List[str]) -> List[QuestionData]:
        """
        Read questions from Excel file tabs
        Args:
            tab_names: List of tab names to read from
        Returns:
            List of QuestionData objects
        """
        questions = []
        try:
            for tab_name in tab_names:
                logger.info(f"Reading tab: {tab_name}")
                df = pd.read_excel(question_path, sheet_name=tab_name)
                for index, row in df.iterrows():
                    question_data = QuestionData(
                        question_id=f"{tab_name}_{index + 1}",
                        question=str(row.get('Question', '')),
                        answer_correct=str(row.get('Answer', '')),
                        tab_name=tab_name,
                        row_index=index + 1
                    )
                    questions.append(question_data)
            logger.info(f"Read {len(questions)} questions from {len(tab_names)} tabs")
            return questions
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    def create_direct_code_prompt(self, question: str, data_files_info: str) -> str:
        """
        Create the direct code prompt for the LLM
        """

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a direct-code SQL generator. Generate complete SQLite code that fulfills user requests without syntax errors.

<|start_header_id|>task<|end_header_id|>

GENERATE ONLY SQLite CODE - NO EXPLANATIONS

<|start_header_id|>requirements<|end_header_id|>

- SQLite compatible syntax only
- Complete code that fulfills ALL user requirements
- No syntax errors
- Code must execute successfully
- Quote column names with spaces: "Column Name"
- Use proper SQLite functions and operators

<|start_header_id|>user_request<|end_header_id|>

{question}

<|start_header_id|>data_files<|end_header_id|>

{data_files_info}

<|start_header_id|>output<|end_header_id|>

Provide ONLY the SQLite code solution:

<|eot_id|>"""
        
#         prompt = f"""Direct-Code LLM

# ### Instruction:
# You are an artificial intelligence assistant. Given some data access methods and a user request, you should write a complete SQOLITE code to fulfill the user's request. Your code must completely fulfill all the user's requirements without syntax errors!

# ### User Request:
# {question}

# ### Data files:
# {data_files_info}

# Please solve the request by SQLITE Code. Provide only the code solution without explanations unless specifically asked for explanations in the question.
# """
        return prompt
    
    async def process_question(self, question_data: QuestionData, data_files_info: str, db_path: str) -> Dict[str, Any]:
        """
        Process a single question using the direct code approach, generate SQL, execute it, and save the result.
        """
        try:
            prompt = self.create_direct_code_prompt(question_data.question, data_files_info)
            start_time = time.time()
            # Use LangChain's ChatOpenAI for LLM call (traced in LangSmith)
            response_message = self.llm.invoke(prompt)
            processing_time = time.time() - start_time

            agent_answer = ""
            if response_message and hasattr(response_message, 'content'):
                agent_answer = response_message.content

            # Extract SQL code block (first code block)
            sql_code = self.extract_sql_code(agent_answer)
            if not sql_code:
                sql_result = "ERROR: No SQL code found in LLM response."
            else:
                # Run the SQL code using run_sqlite_query
                try:
                    sql_result = await run_sqlite_query(sql_code, db_path, markdown=True)
                except Exception as exec_err:
                    sql_result = f"ERROR executing SQL: {exec_err}"
            time.sleep(3)
            cost, total_tokens , input_tokens, output_tokens = self.get_latest_langsmith_cost_and_tokens()
            if cost is None or total_tokens is None:
                total_tokens = 0
                input_tokens = 0
                output_tokens = 0
                cost = 0.0
            result = {
                'question_id': question_data.question_id,
                'question': question_data.question,
                'answer_correct': question_data.answer_correct,
                'agent_final_answer': sql_result,
                'executed_sql': sql_code,
                'intermediate_response': agent_answer,
                'time_seconds': round(processing_time, 2),
                'cost_estimate': cost,
                'num_total_tokens': total_tokens,
                'num_input_tokens': input_tokens,
                'num_output_tokens': output_tokens,
                'tab_name': question_data.tab_name,
                'row_index': question_data.row_index,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Completed question {question_data.question_id} in {processing_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error processing question {question_data.question_id}: {e}")
            return {
                'question_id': question_data.question_id,
                'question': question_data.question,
                'answer_correct': question_data.answer_correct,
                'agent_final_answer': f"ERROR: {str(e)}",
                'executed_sql': None,
                'intermediate_response': agent_answer,
                'time_seconds': round(time.time() - start_time, 2),
                'cost_estimate': 0.0,
                'num_total_tokens': 0,
                'num_input_tokens': 0,
                'num_output_tokens': 0,
                'tab_name': question_data.tab_name,
                'row_index': question_data.row_index,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def extract_sql_code(self, llm_response: str) -> str:
        """
        Extract the first SQL code block from the LLM response.
        """
        # # Look for triple backtick code block with or without 'sql' after the backticks
        # match = re.search(r"```(?:sqlite)?\s*([\s\S]+?)```", llm_response, re.IGNORECASE)
        # if match:
        #     return match.group(1).strip()
        match = re.search(r"```(?:sql)?\s*([\s\S]+?)```", llm_response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: try to find a line starting with SELECT/INSERT/UPDATE/DELETE
        lines = llm_response.splitlines()
        sql_lines = [line for line in lines if line.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE"))]
        return "\n".join(sql_lines).strip() if sql_lines else None
    
    def get_latest_langsmith_cost_and_tokens(self ) -> tuple[Optional[float], Optional[int]]:
        """
        Fetch the latest run from LangSmith and return cost and token usage.
        """
        try:
            if not self.langsmith_client:
                return None, None
                
            project_name = os.environ.get("LANGSMITH_PROJECT")
            if not project_name:
                logger.warning("LANGSMITH_PROJECT environment variable not set.")
                return None, None
                
            runs = self.langsmith_client.list_runs(project_name=project_name, limit=1)
            run = next(runs, None)
            if run:
                return run.total_cost, run.total_tokens , run.input_tokens, run.output_tokens
            else:
                logger.warning("No LangSmith run found.")
                return None, None, None, None
                
        except Exception as e:
            logger.error(f"Error fetching LangSmith run: {e}")
            return None, None, None, None
    

    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save results to Excel file
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    async def process_all_questions(self, question_path: str, data_path: str, tab_names: List[str], output_file: str, 
                            data_files_info: str, file_type: str = "csv"):
        """
        Process all questions from the file and save results
        """
        try:
            # Load questions
            questions = self.read_excel_questions(question_path, tab_names)
            logger.info(f"Processing {len(questions)} questions...")
            results = []
            for i, question_data in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question_data.question_id}")
                result = await self.process_question(question_data, data_files_info, data_path)
                results.append(result)
                logger.info(f"Result: {result['agent_final_answer']}")
                
                if i > 2:
                    break
            self.save_results(results, output_file)
            total_cost = sum(r.get('cost_estimate', 0) for r in results)
            total_tokens = sum(r.get('num_tokens', 0) for r in results)
            total_time = sum(r.get('time_seconds', 0) for r in results)
            logger.info(f"""
Processing Complete!
- Total questions: {len(questions)}
- Total cost: ${total_cost:.4f}
- Total tokens: {total_tokens}
- Total time: {total_time:.2f}s
- Results saved to: {output_file}
            """)
        except Exception as e:
            logger.error(f"Error in process_all_questions: {e}")
            raise

def main():
    """
    Main function to run the Direct-Code LLM processor
    """
    DATA_IDX = 1
    DATA_OPTIONS = ["ai4i2020" , "manufacturing_6G_dataset"]
    DATA_NAME = DATA_OPTIONS[DATA_IDX]
    TAB_NAMES= [DATA_NAME]
    # MODEL_NAME = "gpt-3.5-turbo"
    # MODEL_NAME =  "gemini-2.0-flash"
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    OUTPUT_FILE = f"results/question_results_direct_{DATA_NAME}_{MODEL_NAME.split('-')[0]}.xlsx"
    DATA_FILES = {
        "ai4i2020" : "data/ai4i2020.db",
        "manufacturing_6G_dataset" : "data/manufacturing_6G_dataset.db",
    }
    QUESTION_FILE = r"..\codes\my_QA_dataset.xlsx"
    DATA_FILES_INFOS ={
        "ai4i2020":
    r"""
Available data files and information: The dataset is synthetic and contains 10,000 rows with 14 features each, including a unique identifier, product ID indicating quality variants, temperature readings, rotational speed, torque values, tool wear time, and a label for machine failure caused by one of five independent failure modes.
            CREATE TABLE "ai4i2020" (
                                        "UDI" INTEGER,
                                        "Product ID" TEXT,
                                        "Type" TEXT,
                                        "Air temperature [K]" REAL,
                                        "Process temperature [K]" REAL,
                                        "Rotational speed [rpm]" INTEGER,
                                        "Torque [Nm]" REAL,
                                        "Tool wear [min]" INTEGER,
                                        "Machine failure" INTEGER,
                                        "TWF" INTEGER,
                                        "HDF" INTEGER,
                                        "PWF" INTEGER,
                                        "OSF" INTEGER,
                                        "RNF" INTEGER
                                        )
""",
        "manufacturing_6G_dataset":
    r"""
Available data files and information:
- manufacturing_6G_dataset: The Intelligent Manufacturing Dataset for Predictive Optimization provides sensor data from industrial machines, 6G network performance metrics, and production efficiency indicators. It includes a target column \"Efficiency_Status\" classifying manufacturing efficiency as High, Medium, or Low for applications such as AI-based predictive maintenance and deep learning model training.
            CREATE TABLE "manufacturing_6G_dataset" (
                                        "Timestamp" TEXT,
                                        "Machine_ID" INTEGER,
                                        "Operation_Mode" TEXT,
                                        "Temperature_C" REAL,
                                        "Vibration_Hz" REAL,
                                        "Power_Consumption_kW" REAL,
                                        "Network_Latency_ms" REAL,
                                        "Packet_Loss_%" REAL,
                                        "Quality_Control_Defect_Rate_%" REAL,
                                        "Production_Speed_units_per_hr" REAL,
                                        "Predictive_Maintenance_Score" REAL,
                                        "Error_Rate_%" REAL,
                                        "Efficiency_Status" TEXT
                                        )
"""
    }
    processor = DirectCodeLLMProcessor(model_name=MODEL_NAME)
    asyncio.run(processor.process_all_questions(
        question_path=QUESTION_FILE,
        data_path=DATA_FILES[DATA_NAME],
        tab_names=TAB_NAMES,
        output_file=OUTPUT_FILE,
        data_files_info=DATA_FILES_INFOS[DATA_NAME],
        file_type="excel"
    ))

if __name__ == "__main__":
    main()