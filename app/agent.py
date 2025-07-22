import streamlit as st
import asyncio
from dotenv import load_dotenv
import logging
import json
from plotly.graph_objs import Figure
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load environment variables
load_dotenv(".env")

from .utils import generate_sqlite_table_info_query, format_table_info
from .tools import tool_schema_defintion, run_sqlite_query, plot_chart, create_plotly_figure
from .bot import LangGraphChatBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Data Analysis Chatbot", layout="wide")

async def create_bot(table_info: str , db_path: str, model: str):
    """Create and return a LangGraphChatBot instance"""

    # System message for llama
    llama_system_message = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a manufacturing data analyst. Convert database queries to business insights for non-technical users.

<|start_header_id|>instructions<|end_header_id|>

**CORE RULES:**
- ONLY respond to data analysis requests using provided schema
- Decline unrelated requests politely  
- NO technical details (SQL, table names) in responses
- Business language only
- Execute tools promptly but handle errors gracefully
- Maximum 2 tool attempts per request - then stop and explain

**TOOLS - EXECUTE ONCE ONLY:**
- `query_db`: Data extraction, filtering, calculations, aggregations
- `plot_chart`: All visualizations (graphs, charts, plots)
- If tool fails: Fix SQL syntax and retry ONCE only
- If second attempt fails: Explain limitation and stop

**SQL REQUIREMENTS:**
- SQLite compatible only
- Quote columns: `"Column Name"`
- Max 5 results for top N, max 10 for general queries
- Show 5-8 relevant columns only
- NEVER use `SELECT * FROM table`
- Always use WHERE clauses to limit data

<|start_header_id|>correlation_triggers<|end_header_id|>

**AUTOMATICALLY CALCULATE CORRELATION when user asks about:**
- "impact of X on Y"
- "influence of X on Y" 
- "relationship between X and Y"
- "how X affects Y"
- "correlation between X and Y"
- "does X influence Y"
- "connection between X and Y"

**For these questions: ALWAYS run correlation analysis first, then explain business meaning**

<|start_header_id|>statistical_formulas<|end_header_id|>

**Correlation (Pearson) - USE FOR IMPACT/INFLUENCE QUESTIONS:**
```sql
SELECT 
    (COUNT(*) * SUM("Col1" * "Col2") - SUM("Col1") * SUM("Col2")) / 
    SQRT((COUNT(*) * SUM("Col1" * "Col1") - SUM("Col1") * SUM("Col1")) * 
         (COUNT(*) * SUM("Col2" * "Col2") - SUM("Col2") * SUM("Col2"))) AS correlation,
    COUNT(*) as sample_size,
    '"Col1"' as feature_1,
    '"Col2"' as feature_2
FROM table_name
```

**Standard Deviation:**
```sql
SQRT((SUM("Col" * "Col") - COUNT(*) * AVG("Col") * AVG("Col")) / (COUNT(*) - 1))
```

**MEDIAN - CRITICAL:**
- NEVER use AVG() for median (that's mean)
- Use ROW_NUMBER() window function for true median

**Correlation Interpretation:**
- Strong relationship: >0.7 or <-0.7
- Moderate relationship: 0.3-0.7 or -0.3--0.7
- Weak relationship: <0.3 and >-0.3
- Positive: As X increases, Y increases
- Negative: As X increases, Y decreases

<|start_header_id|>response_format<|end_header_id|>

- Rich Markdown with tables
- Clear business insights
- Explain assumptions
- Ask clarifying questions when needed

<|start_header_id|>workflow<|end_header_id|>

**CORRELATION WORKFLOW FOR IMPACT/INFLUENCE QUESTIONS:**
1. User asks about impact/influence/relationship → Immediately calculate correlation
2. Execute correlation SQL query using appropriate columns
3. Interpret correlation coefficient in business terms
4. Explain what the relationship means for manufacturing operations
5. Provide actionable insights based on correlation strength

**GENERAL WORKFLOW:**
1. User requests data → Execute `query_db` tool
2. If query succeeds → Present business insights
3. If query fails → Fix SQL and retry ONCE
4. If second attempt fails → Stop and explain limitation
5. Never attempt same query more than twice

**STOP CONDITIONS:**
- After successful query execution
- After 2 failed attempts
- When user changes topic

<|start_header_id|>schema<|end_header_id|>

{table_info}

<|eot_id|>"""

    # System message for gpt and gemini
    gpt_gemini_system_message = f"""You are a manufacturing data analyst. Transform database queries into business insights for non-technical users.

    **Core Rules:**
    - Only respond to data analysis requests using the provided schema
    - If unrelated, politely decline
    - No technical details (SQL, table/column names) in responses
    - Business-friendly language only

    **Tool Usage:**
    - Use `query_db` for data extraction, filtering, calculations, and aggregations
    - Use `plot_chart` for visualizations, graphs, charts, and plotting requests
    - Always use appropriate tools when users ask for data or visualizations

    **Correlation & Feature Analysis:**
    - For correlation queries: Calculate Pearson correlation coefficient using SQL formulas
    - Use SQL functions: `AVG()`, `SUM()`, `COUNT()`, `SQRT()` for correlation calculations
    - Formula: `(n*Σxy - Σx*Σy) / SQRT((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))`
    - **SQL SYNTAX**: Always properly quote column names: `"Column Name"` not `"Column Name`
    - For x²: use `"Column Name" * "Column Name"` with proper quotes on both sides
    - **STANDARD DEVIATION**: Use formula `SQRT((SUM("Column" * "Column") - COUNT(*) * AVG("Column") * AVG("Column")) / (COUNT(*) - 1))`
    - **VARIANCE**: Use formula `(SUM("Column" * "Column") - COUNT(*) * AVG("Column") * AVG("Column")) / (COUNT(*) - 1)`
    - **CORRELATION EXAMPLE**: 
  ```sql
  SELECT (COUNT(*) * SUM("Col1" * "Col2") - SUM("Col1") * SUM("Col2")) / 
         SQRT((COUNT(*) * SUM("Col1" * "Col1") - SUM("Col1") * SUM("Col1")) * 
              (COUNT(*) * SUM("Col2" * "Col2") - SUM("Col2") * SUM("Col2"))) AS correlation
  FROM table_name ```
    - Calculate covariance, variance, and standard deviation within SQL queries
    - For feature impact: Use SQL aggregations to compare means, medians across groups
    - Explain correlation strength: >0.7 strong, 0.3-0.7 moderate, <0.3 weak
    - **CRITICAL**: When user asks for median, calculate it using SQL percentile functions or ROW_NUMBER() - NEVER return mean instead of median

    **Data Limits:**
    - Top N queries: max 5 results
    - All records: max 10 results  
    - Show 5-8 most relevant columns
    - Format dates/numbers as readable strings in SQL
    - **STRICTLY FORBIDDEN**: Never use `SELECT * FROM table` or show all table data - always use specific columns and WHERE clauses to limit results

    **Query Requirements:**
    - SQLite compatible only
    - Handle case variations and unknown enum values
    - Use only schema-defined tables/columns
    - If query fails, retry with corrections
    - **MEDIAN CALCULATION**: Never use AVG() for median - use this SQL pattern:
    - **WRONG**: `SELECT AVG(column) FROM table` - this is MEAN not MEDIAN
    - **CORRECT**: Use ROW_NUMBER() to find middle value(s) then average those

    **Response Format:**
    - Rich Markdown with tables
    - Clear business insights
    - Explain assumptions clearly
    - Ask clarifying questions when needed

    **Schema:**
    {table_info}"""

    # Choose system message based on model name
    model_lower = model.lower()
    if "llama" in model_lower or "meta" in model_lower:
        system_message = llama_system_message
    elif "gpt" in model_lower or "gemini" in model_lower:
        system_message = gpt_gemini_system_message
    else:
        system_message = llama_system_message  # Default fallback

    tool_functions = {
        "query_db": run_sqlite_query,
        "plot_chart": plot_chart
    }

    tools_schema = tool_schema_defintion(db_path)

    return LangGraphChatBot(system_message, tools_schema, tool_functions, model)

async def process_message(bot, message: str, max_iterations: int = 5):
    """Process a message through the bot with tool calling support"""
    responses = []
    charts = []
    
    logger.info(f"Processing message: {message}")
    # Get initial response
    try:
        response_message, steps = await bot(message)
    except Exception as e:
        logger.error(f"Error during bot message processing: {e}")
        return responses, charts

    if response_message.content:
        responses.append(response_message.content)
        logger.info("Appended assistant response to responses list.")

    # Process steps to extract chart data
    for msg in steps:
        logger.debug(f"Processing step message of type: {type(msg)}")
        if isinstance(msg, ToolMessage):
            logger.debug('Tool message found!')
            if hasattr(msg, 'name') and msg.name == "plot_chart":
                logger.info("Plot chart tool message found!")
                try:
                    chart_data = json.loads(msg.content)
                    fig = create_plotly_figure(chart_data)
                    charts.append(fig)
                    logger.info("Chart created and added to charts list.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing chart data: {e}")
                except Exception as e:
                    logger.error(f"Error creating chart: {e}")

    return responses, charts