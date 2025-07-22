"""
Configuration file for Excel Question Processor
"""

# Excel file configuration
EXCEL_PATH = r"QA/my_QA_dataset.xlsx"

# Excel tab names to process
# TAB_NAMES = ["ai4i2020"]  
# DB_PATH = "data/ai4i2020.db"  
# OUTPUT_FILE = "results/question_results_ai4i2020.xlsx"

TAB_NAMES = ["manufacturing_6G_dataset"]
DB_PATH = "data/manufacturing_6G_dataset.db" 
OUTPUT_FILE = "results/question_results_manufacturing_6G_dataset.xlsx"

# Model configuration
# MODEL_NAME = "gemini-2.0-flash"  
MODEL_NAME = "gpt-3.5-turbo"

BATCH_SIZE = 5  
MAX_RETRIES = 3  
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 