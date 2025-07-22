"""
Configuration file for Excel Question Processor
"""

# Excel file configuration
EXCEL_PATH = r"C:\Users\nmofa\OneDrive - Danmarks Tekniske Universitet\Desktop\dtu\my_research\cirpe_2025\codes\my_QA_dataset.xlsx"

# Excel tab names to process
# TAB_NAMES = ["ai4i2020"]  # Adjust to your actual tab names
# DB_PATH = "data/ai4i2020.db"  # Adjust to your database path
# OUTPUT_FILE = "results/question_results_ai4i2020.xlsx"

TAB_NAMES = ["manufacturing_6G_dataset"]
DB_PATH = "data/manufacturing_6G_dataset.db"  # Adjust to your database path
OUTPUT_FILE = "results/question_results_manufacturing_6G_dataset.xlsx"

# Model configuration
# MODEL_NAME = "gemini-2.0-flash"  
MODEL_NAME = "gpt-3.5-turbo"


# Processing configuration
BATCH_SIZE = 5  # Number of questions to process before logging progress
MAX_RETRIES = 3  # Maximum number of retries for failed questions


# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 