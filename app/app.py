import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import json
from csv_to_db import csv_to_sqlite  # <-- Add this import
from langchain_openai import ChatOpenAI  # <-- Add for summarization
import sqlite3  # <-- Add for SQLite schema queries
import asyncio
from agent import create_bot, process_message
from utils import generate_sqlite_table_info_query, run_db_query
from plotly.graph_objs import Figure
# from agent import AgentState, agent_loop
import ast
import plotly.graph_objects as go

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Add custom CSS for manufacturing theme
st.set_page_config(page_title="Agentic Manufacturing Data Analysis", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@700&family=Roboto+Slab:wght@700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Update title and tabs with manufacturing emojis/icons
st.title("\U0001F3ED Agentic Manufacturing Data Analysis")

tabs = st.tabs([
    "\U0001F4C2 Data Management",
    "\U0001F916 Q&A"
])

# Ensure 'data' directory and data_dictionary.json exist
os.makedirs("data", exist_ok=True)
data_dict_path = os.path.join("data", "data_dictionary.json")
if not os.path.exists(data_dict_path):
    with open(data_dict_path, "w") as f:
        json.dump({}, f)

with open(data_dict_path, "r") as f:
    data_dictionary = json.load(f)

# Add a helper function for summarization
def summarize_text(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    prompt = f"Summarize the following dataset description in 1-2 sentences for a data analysis agent.\n\nDescription: {text}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes dataset descriptions."},
        {"role": "user", "content": prompt}
    ]
    try:
        summary = llm.invoke(messages)
        return summary.content.strip()
    except Exception as e:
        return "(Summary unavailable)"

# Helper to get table schema info from SQLite .db
def get_sqlite_table_info(db_path):
    sql_query = """SELECT sql FROM sqlite_master m WHERE m.type='table' AND m.name NOT LIKE 'sqlite_%';"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql_query)
        result = cur.fetchall()
        conn.close()
        # Format as string
        table_info = '\n'.join([item[0] for item in result if item[0]])
        return table_info
    except Exception as e:
        return f"(Could not retrieve table info: {e})"

# Helper function to get comprehensive table info for the agent
async def get_comprehensive_table_info(db_path):
    """Get comprehensive table information including schema and sample data"""
    try:
        # Get basic schema info
        schema_query = generate_sqlite_table_info_query([])
        schema_results, schema_columns = await run_db_query(schema_query, db_path)
        
        table_info = ""
        if schema_results:
            table_info = '\n'.join([item[0] for item in schema_results if item[0]])
        
        # Get table names for sample data
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        tables_results, _ = await run_db_query(tables_query, db_path)
        
        # Add sample data for each table
        for table_row in tables_results:
            table_name = table_row[0]
            sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
            sample_results, sample_columns = await run_db_query(sample_query, db_path)
            
            if sample_results:
                table_info += f"\n\nSample data from {table_name}:\n"
                table_info += f"Columns: {', '.join(sample_columns)}\n"
                for row in sample_results:
                    table_info += f"Sample row: {row}\n"
        
        return table_info
    except Exception as e:
        return f"Error getting table info: {e}"

# Define available models and allow user to select
MODEL_OPTIONS = [
    "gpt-3.5-turbo",
    "gemini-2.0-flash",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
]

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose LLM model for the agent:", MODEL_OPTIONS, index=0)

# --- Tab 1: Data Management ---
with tabs[0]:
    st.header("\U0001F4C2 Upload and Manage Manufacturing Data")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            # Convert CSV to .db and save
            base_name = os.path.splitext(file.name)[0]
            db_path = os.path.join("data", f"{base_name}.db")
            try:
                csv_to_sqlite(file_path, db_path, table_name=base_name)
            except Exception as e:
                st.error(f"Failed to convert {file.name} to .db: {str(e)}")
            # Prompt for description
            description = st.text_area(
                f"Enter a description for {file.name}",
                value="",
                key=f"desc_upload_{file.name}",
                help="Provide a detailed description of this dataset",
                height=200
            )
            if file.name not in data_dictionary:
                data_dictionary[file.name] = {}
            data_dictionary[file.name]["description_full"] = description
            # Summarize and store, including table info
            summary = summarize_text(description) if description else ""
            table_info = get_sqlite_table_info(db_path)
            combined_summary = summary + (f"\n\nTable schema info:\n{table_info}" if table_info else "")
            data_dictionary[file.name]["description_summary"] = combined_summary
        with open(data_dict_path, "w") as f:
            json.dump(data_dictionary, f, indent=4)
        st.success("Files uploaded, converted to .db, and descriptions summarized!")

    # List available CSV files
    available_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    selected_files = st.multiselect(
        "Select files to analyze",
        available_files,
        key="selected_files"
    )
    new_descriptions = {}
    if selected_files:
        file_tabs = st.tabs(selected_files)
        for tab, filename in zip(file_tabs, selected_files):
            with tab:
                try:
                    df = pd.read_csv(os.path.join("data", filename))
                    st.write(f"Preview of {filename}:")
                    st.dataframe(df.head())
                    st.subheader("Dataset Information")
                    desc_full = data_dictionary.get(filename, {}).get('description_full', '')
                    desc_summary = data_dictionary.get(filename, {}).get('description_summary', '')
                    # Editable full description, capture edits
                    new_descriptions[filename] = st.text_area(
                        "Full Description (editable)",
                        value=desc_full,
                        key=f"description_full_{filename}",
                        help="Complete version of the dataset description",
                        height=200
                    )
                    st.text_area(
                        "Summarized Description (used for analysis)",
                        value=desc_summary,
                        key=f"description_summary_{filename}",
                        help="Short summary for the agent (auto-generated)",
                        height=100,
                        disabled=True
                    )
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
        if st.button("Save Descriptions"):
            for filename, description in new_descriptions.items():
                if description:
                    if filename not in data_dictionary:
                        data_dictionary[filename] = {}
                    data_dictionary[filename]['description_full'] = description
                    # Regenerate summary from edited full description and update table info
                    base_name = os.path.splitext(filename)[0]
                    db_path = os.path.join("data", f"{base_name}.db")
                    summary = summarize_text(description)
                    table_info = get_sqlite_table_info(db_path)
                    data_dictionary[filename]['description_summary'] = summary
                    data_dictionary[filename]['table_info'] = table_info
            with open(data_dict_path, "w") as f:
                json.dump(data_dictionary, f, indent=4)
            st.success("Descriptions saved and summarized successfully!")
    else:
        st.info("No CSV files available or selected. Please upload and select files.")

# --- Tab 2: Q&A ---
with tabs[1]:
    st.header(":factory: Manufacturing Data Chatbot")
    st.markdown("<style>div[data-testid='stChatMessage']{background: #f0f4f8; border-radius: 12px; margin-bottom: 8px;} .user-msg{color:#1565c0;} .bot-msg{color:#388e3c;} .code-block{background:#e3eaf2; border-radius:8px; padding:8px; font-family:monospace;}</style>", unsafe_allow_html=True)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "intermediate_steps_store" not in st.session_state:
        st.session_state["intermediate_steps_store"] = []
    if "bot_instance" not in st.session_state:
        st.session_state["bot_instance"] = None
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = selected_model
    
    # Only allow chat if files are selected
    selected_files = st.session_state.get("selected_files", [])
    if not selected_files:
        st.warning("Please select at least one dataset in the 'Data Management' tab.")
    else:
        # For simplicity, use the first selected file for the agent
        filename = selected_files[0]
        base_name = os.path.splitext(filename)[0]
        db_path = os.path.join("data", f"{base_name}.db")
        
        # Initialize bot if not exists or if file/model changed
        if (st.session_state["bot_instance"] is None or 
            st.session_state.get("current_file") != filename or
            st.session_state.get("current_model") != selected_model):
            
            with st.spinner("Initializing agent..."):
                # Get comprehensive table info
                table_info = asyncio.run(get_comprehensive_table_info(db_path))
                
                # Create bot instance with selected model
                st.session_state["bot_instance"] = asyncio.run(create_bot(table_info , db_path, selected_model))
                st.session_state["current_file"] = filename
                st.session_state["current_model"] = selected_model
                st.success(f"Agent initialized for {filename} using model {selected_model}")
        
        # Display chat history
        for idx, msg in enumerate(st.session_state["chat_history"]):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(f"<span class='user-msg'><b>You:</b> {msg['content']}</span>", unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"<span class='bot-msg'><b>Agent:</b> {msg['content']}</span>", unsafe_allow_html=True)
                    
                    # Display any charts - charts are already Figure objects
                    if msg.get("charts"):
                        for chart in msg["charts"]:
                            if isinstance(chart, Figure):
                                st.plotly_chart(chart, use_container_width=True)
                            else:
                                st.error(f"Invalid chart type: {type(chart)}")

        
        # Handle user input
        user_input = st.chat_input("Ask a question about your manufacturing data...")
        if user_input and st.session_state["bot_instance"]:
            # Add user message to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(f"<span class='user-msg'><b>You:</b> {user_input}</span>", unsafe_allow_html=True)
            
            # Process message with agent
            with st.spinner("Agent is thinking..."):
                try:
                    # Process the message through the agent
                    responses, charts = asyncio.run(
                        process_message(st.session_state["bot_instance"], user_input)
                    )
                    
                    # Combine all responses
                    full_response = "\n\n".join(responses) if responses else "I'm sorry, I couldn't process your request."
                    
                    # Add assistant response to chat history
                    assistant_msg = {
                        "role": "assistant", 
                        "content": full_response,
                        "charts": charts if charts else None
                    }
                    st.session_state["chat_history"].append(assistant_msg)
                    
                    # Display assistant response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"<span class='bot-msg'><b>Agent:</b> {full_response}</span>", unsafe_allow_html=True)
                        
                        # Display any charts - charts are already Figure objects
                        if charts:
                            for chart in charts:
                                print("Displaying chart:")
                                
                                # chart is already a Plotly Figure object, no need to eval
                                if isinstance(chart, Figure):
                                    st.plotly_chart(chart, use_container_width=True)
                                else:
                                    st.error(f"Invalid chart type: {type(chart)}")

                except Exception as e:
                    error_msg = f"Error processing your request: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error to chat history
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.rerun()