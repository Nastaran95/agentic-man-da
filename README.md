# Agentic Manufacturing Analysis

A modular agentic AI system for natural language-driven data analysis in manufacturing using LLMs and LangGraph.

## Directory Structure

- **app/**: Main application code (Streamlit UI, agent logic, tools, utils, etc.)
- **experiments/**: Scripts for batch evaluation, direct LLM code, and research experiments
- **results/**: Output files and evaluation results
- **QA/**: The created questions and answers for the utilized datasets. 

## Running the App

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```
3. **App UI Example:**
   ![App Screenshot](assets/Screenshot%202025-07-09%20150015.png)

## Using Different LLMs and Datasets
- Select your preferred LLM (Llama, GPT, Gemini) from the sidebar dropdown.
- Upload or select a dataset (CSV) in the Data Management tab.
- The agent will use the selected LLM and dataset for all Q&A in the chat tab.

## Downloading Datasets from Kaggle
1. Find your dataset on Kaggle 
    * [AI4I2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)
    * [Man6Gdata](https://www.kaggle.com/datasets/ziya07/intelligent-manufacturing-dataset)
2. Download the CSV file(s).
3. Then you can select/upload these files.

