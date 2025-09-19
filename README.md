# Agentic Manufacturing Analysis

A modular agentic AI system for natural language-driven data analysis in manufacturing using LLMs and LangGraph.

> **Repository for the accepted paper:**  
> **Title:** Agentic Data Analysis for Intelligent Manufacturing: Benchmark-Driven Evaluation of Agentic vs. Direct LLM Approaches  
> **Conference:** CIRPe 2025

**Abstract:**  
Recently, agentic artificial Intelligence (AI) has gained strong attention, showing promising results in domains such as quality control, knowledge management, and cost optimization. Yet, its application within manufacturing remains largely underexplored. Moreover, data analysis is a critical step across many manufacturing processes, but interpreting data often depends heavily on the expertise of technical professionals. To address this gap, this paper introduces a lightweight, agentic framework that enables non-expert users to interact with manufacturing datasets through natural language. The system integrates language models (LLMs) with modular tool orchestration to support data querying, analysis, and visualization via conversational interfaces. The system is evaluated using two representative manufacturing datasets and a benchmark of structured natural language queries inspired by TableBench. Comparative results across multiple LLMs reveal that our agentic approach outperforms direct prompting methods in both accuracy and interpretability. These findings demonstrate the feasibility and effectiveness of deploying agentic AI systems for real-world industrial data analysis and point toward more accessible and scalable AI-driven manufacturing solutions.

[Read the paper (PDF)](PROCIR_CIRPe_2025_revised_final.pdf)

## Directory Structure

- **app/**: Main application code (Streamlit UI, agent logic, tools, utils, etc.)
- **experiments/**: Scripts for evaluation, direct LLM code, and research experiments
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
1. Find the utilized dataset on Kaggle 
    * [AI4I2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/data)
    * [Man6Gdata](https://www.kaggle.com/datasets/ziya07/intelligent-manufacturing-dataset)
2. Download the CSV file(s).
3. Then you can select/upload these files.
4. You can also use your own dataset. 

## 📖 Citing This Work

If you use this repository or build upon our work, please cite the following paper:

```bibtex
@inproceedings{moradzadeh2025agentic,
  title={Agentic Data Analysis for Intelligent Manufacturing: Benchmark-Driven Evaluation of Agentic vs. Direct LLM Approaches},
  author={Moradzadeh Farid, Nastaran and Taghizadeh, Alireza and Shafiee, Sara},
  booktitle={Proceedings of the 13th CIRP Global Web Conference (CIRPe 2025)},
  year={2025}
}
```