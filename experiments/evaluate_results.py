import os
import glob
import pandas as pd
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Settings
RESULTS_DIR = "results"
OUTPUT_FILE = "final_evaluation_meta.xlsx"
DATASETS = {
    "ai4i2020": [
        "question_results_ai4i2020_openai.xlsx",
        "question_results_ai4i2020_gemini.xlsx",
        "question_results_direct_ai4i2020_gpt.xlsx",
        "question_results_direct_ai4i2020_gemini.xlsx",
        "question_results_ai4i2020_meta.xlsx",
        "question_results_direct_ai4i2020_meta.xlsx",
    ],
    "manufacturing_6G_dataset": [
        "question_results_manufacturing_6G_dataset_openai.xlsx",
        "question_results_manufacturing_6G_dataset_gemini.xlsx",
        "question_results_direct_manufacturing_6G_dataset_gpt.xlsx",
        "question_results_direct_manufacturing_6G_dataset_gemini.xlsx",
        "question_results_manufacturing_6G_dataset_meta.xlsx",
        "question_results_direct_manufacturing_6G_dataset_meta.xlsx",
    ],
}
METHODS = ["gpt", "gemini", "directgpt", "directgemini", "meta", "directmeta"]

def get_method_from_filename(filename):
    if "direct" in filename and "gpt" in filename:
        return "directgpt"
    if "direct" in filename and "gemini" in filename:
        return "directgemini"
    if "direct" in filename and "meta" in filename:
        return "directmeta"
    if "openai" in filename or "gpt" in filename:
        return "gpt"
    if "gemini" in filename:
        return "gemini"
    if "meta" in filename:
        return "meta"
    return None

def get_dataset_from_filename(filename):
    if "ai4i2020" in filename:
        return "ai4i2020"
    if "manufacturing_6G_dataset" in filename or "man6gdata" in filename:
        return "manufacturing_6G_dataset"
    return None

def build_eval_prompt(question, correct_answer, agent_answer):
    return f"""
You are an expert evaluator. Given the following:
- Question: {question}
- Correct Answer: {correct_answer}
- Agent Answer: {agent_answer}

Evaluate the agent's answer using two approaches:

1. Strict (score: 0 or 1): Does the agent's answer exactly match the correct answer in content, format, and precision?

2. Flexible (score: 0 or 1): Does the agent's answer convey the same core meaning as the correct answer? Award 1 only if:
   - The factual content is accurate and substantively equivalent
   - All key information from the correct answer is present (even if reworded)
   - The agent actually addresses the question rather than refusing to answer
   - Different phrasing/format is acceptable only if the essential meaning and data are preserved

For each approach, provide:
- The score (0 or 1)
- A brief explanation (maximum 2 sentences)

Respond strictly in this format:
approach strict = [0 or 1] Explanation: [your explanation in maximum 2 sentences]
approach flexible = [0 or 1] Explanation: [your explanation in maximum 2 sentences]
"""

def evaluate_answer_gpt(question, correct_answer, agent_answer):
    prompt = build_eval_prompt(question, correct_answer, agent_answer)
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"EVAL_ERROR: {e}"

def parse_eval_response(eval_response):
    # Extract strict and flexible scores and explanations from the new format
    import re
    strict_score, strict_exp, flexible_score, flexible_exp = None, "", None, ""
    strict_match = re.search(r"approach strict\s*=\s*([01])\s*Explanation:\s*(.*?)(?:\n|$)", eval_response, re.IGNORECASE | re.DOTALL)
    flexible_match = re.search(r"approach flexible\s*=\s*([01])\s*Explanation:\s*(.*?)(?:\n|$)", eval_response, re.IGNORECASE | re.DOTALL)
    if strict_match:
        strict_score = int(strict_match.group(1))
        strict_exp = strict_match.group(2).strip()
    if flexible_match:
        flexible_score = int(flexible_match.group(1))
        flexible_exp = flexible_match.group(2).strip()
    return strict_score, strict_exp, flexible_score, flexible_exp

def merge_results_for_dataset(dataset, files):
    # Read all files and merge on question_id
    dfs = {}
    for f in files:
        method = get_method_from_filename(f)
        if not method:
            continue
        df = pd.read_excel(os.path.join(RESULTS_DIR, f))
        # Standardize column names
        df = df.rename(columns={
            "agent_final_answer": f"answer_{method}",
            "Agent Final Answer": f"answer_{method}",
            "question": "question",
            "Question": "question",
            "answer_correct": "answer_correct",
            "Answer": "answer_correct",
            "question_id": "question_id",
            "Question ID": "question_id",
        })
        # Find the correct token column name
        token_col = None
        for col in ["num_tokens", "num_total_tokens"]:
            if col in df.columns:
                token_col = col
                break
        # Add time and token columns with method-specific names
        cols = ["question_id", "question", "answer_correct", f"answer_{method}"]
        if "time_seconds" in df.columns:
            df[f"time_{method}"] = df["time_seconds"]
            cols.append(f"time_{method}")
        if token_col:
            df[f"tokens_{method}"] = df[token_col]
            cols.append(f"tokens_{method}")
        dfs[method] = df[cols]
    # Merge all on question_id
    merged = None
    for method in METHODS:
        if method in dfs:
            if merged is None:
                merged = dfs[method]
            else:
                merged = pd.merge(merged, dfs[method], on=["question_id", "question", "answer_correct"], how="outer")
    return merged

def process_and_evaluate():
    writer = pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl")
    for dataset, files in DATASETS.items():
        merged = merge_results_for_dataset(dataset, files)
        if merged is None:
            print(f"No data for {dataset}")
            continue
        # For each method, evaluate
        for method in METHODS:
            answer_col = f"answer_{method}"
            strict_col = f"{method}_strict_score"
            strict_exp_col = f"{method}_strict_explanation"
            flexible_col = f"{method}_flexible_score"
            flexible_exp_col = f"{method}_flexible_explanation"
            merged[strict_col] = None
            merged[strict_exp_col] = ""
            merged[flexible_col] = None
            merged[flexible_exp_col] = ""
            if answer_col in merged:
                for idx, row in tqdm(merged.iterrows(), total=merged.shape[0], desc=f"Evaluating {dataset} {method}"):
                    ans = row[answer_col]
                    if pd.isna(ans) or ans == "":
                        continue
                    eval_response = evaluate_answer_gpt(row["question"], row["answer_correct"], ans)
                    strict_score, strict_exp, flexible_score, flexible_exp = parse_eval_response(eval_response)
                    merged.at[idx, strict_col] = strict_score
                    merged.at[idx, strict_exp_col] = strict_exp
                    merged.at[idx, flexible_col] = flexible_score
                    merged.at[idx, flexible_exp_col] = flexible_exp
                    # if idx>2:
                    #     break
        # Move all time and token columns to the end
        time_token_cols = [col for col in merged.columns if col.startswith('time_') or col.startswith('tokens_')]
        other_cols = [col for col in merged.columns if col not in time_token_cols]
        merged = merged[other_cols + time_token_cols]
        # Write to Excel sheet
        merged.to_excel(writer, sheet_name=dataset, index=False)
    writer.close()
    print(f"Final evaluation saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_and_evaluate() 