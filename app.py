import sqlite3
import torch
import re
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from rapidfuzz import process
import streamlit as st

# Setup Streamlit page
st.set_page_config(page_title="Data Chatbot", page_icon="🤖", layout="wide")
st.title("SQLite Database Chatbot")

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_conn" not in st.session_state:
    st.session_state.db_conn = None
if "schema" not in st.session_state:
    st.session_state.schema = ""
if "column_names" not in st.session_state:
    st.session_state.column_names = []
if "tables" not in st.session_state:
    st.session_state.tables = []

# ============================================================
# 1. DATABASE UPLOAD & SCHEMA PARSING
# ============================================================
with st.sidebar:
    st.header("Upload Database")
    uploaded_file = st.file_uploader("Upload a SQLite file", type=["sqlite", "db", "sqlite3"])
    
    if uploaded_file is not None and st.session_state.db_conn is None:
        # Write to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Connect to DB
        conn = sqlite3.connect(temp_path, check_same_thread=False)
        st.session_state.db_conn = conn
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        st.session_state.tables = tables
        
        schema = ""
        column_names = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            cols = cursor.fetchall()
            schema += f"\nTable: {table}\nColumns: " + ", ".join([f"{c[1]} ({c[2]})" for c in cols])
            column_names.extend([c[1] for c in cols])
            
        st.session_state.schema = schema
        st.session_state.column_names = column_names
        
        st.success(f"Database Loaded! Tables: {', '.join(tables)}")

# ============================================================
# 2. MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading Model... (This may take a minute)")
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We remove device_map="auto" and torch_dtype=torch.float16 to force a safe CPU load
    # This prevents freezing on Windows environments that do not have a dedicated NVIDIA setup 
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Let the user know if a DB is missing
if st.session_state.db_conn is not None:
    tokenizer, model = load_model()

# ============================================================
# 3. CORE TOOLS
# ============================================================
def run_sql_query(sql_query: str):
    sql_query = sql_query.split(';')[0].replace('"}}', '').strip()
    st.markdown(f"`SQL Run: {sql_query}`")
    try:
        return pd.read_sql_query(sql_query, st.session_state.db_conn)
    except Exception as e:
        return f"Error executing query: {e}"

def generate_plot(table_name: str, x_column: str, y_column: str, plot_type: str = 'line'):
    try:
        df = pd.read_sql_query(f'SELECT "{x_column}", "{y_column}" FROM "{table_name}" LIMIT 500', st.session_state.db_conn)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        if plot_type == 'line':
            sns.lineplot(data=df, x=x_column, y=y_column, ax=ax)
        else:
            sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
            
        st.pyplot(fig)
        return "Plot displayed successfully."
    except Exception as e:
        return f"Plot Error: {e}"

def save_table_to_csv(table_name: str):
    best_tab = process.extractOne(table_name, st.session_state.tables, score_cutoff=60)
    if not best_tab:
        return f"Error: Table '{table_name}' not found."

    filename = f"{best_tab[0]}_export.csv"
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{best_tab[0]}"', st.session_state.db_conn)
        csv = df.to_csv(index=False).encode('utf-8')
        st.session_state.download_data = {"data": csv, "filename": filename}
        return f"Prepared Data for download."
    except Exception as e:
        return f"Export Error: {e}"

# ============================================================
# 4. REFINED ROUTER (Fixes SELECT SELECT and Plot detection)
# ============================================================
def get_tool_call(question):
    low_q = question.lower()
    column_names = st.session_state.column_names
    tables = st.session_state.tables

    # 1. Aggregation Keyword Fallback
    mappings = {"average": "AVG", "mean": "AVG", "max": "MAX", "min": "MIN", "sum": "SUM", "count": "COUNT"}
    selected_func = next((func for word, func in mappings.items() if word in low_q), None)

    if selected_func:
        possible_cols = [c for c in column_names if c.lower() in low_q]
        best_tab = process.extractOne(low_q, tables, score_cutoff=60)
        if possible_cols and best_tab:
            actual_col = process.extractOne(low_q, possible_cols)[0]
            sql = f'SELECT {selected_func}("{actual_col}") FROM "{best_tab[0]}"'
            return {"tool_name": "run_sql_query", "arguments": {"sql_query": sql}}

    # 2. Plotting detection
    if any(word in low_q for word in ["plot", "chart", "graph", "visualize"]):
        best_tab = process.extractOne(low_q, tables, score_cutoff=60)
        mentioned_cols = [c for c in column_names if c.lower() in low_q]
        if best_tab and len(mentioned_cols) >= 2:
            return {"tool_name": "generate_plot", "arguments": {
                "table_name": best_tab[0], "x_column": mentioned_cols[1], "y_column": mentioned_cols[0]
            }}

    # 3. Export detection
    if any(word in low_q for word in ["save", "download", "export", "csv"]):
        best_tab = process.extractOne(low_q, tables, score_cutoff=60)
        if best_tab:
            return {"tool_name": "save_table_to_csv", "arguments": {"table_name": best_tab[0]}}

    # 4. LLM Fallback (Cleaned)
    schema = st.session_state.schema
    prompt = f"<|system|>Output SQL for: {schema[:300]}\n<|user|>{question}\n<|assistant|>SELECT "
    
    # Needs torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
    ans = re.sub(r'^select\s+', '', ans, flags=re.IGNORECASE)
    return {"tool_name": "run_sql_query", "arguments": {"sql_query": "SELECT " + ans}}

# ============================================================
# 5. CHAT INTERFACE
# ============================================================
if st.session_state.db_conn is None:
    st.info("Please upload a SQLite database from the sidebar to begin chatting.")
else:
    # Display chat messages history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "dataframe":
                st.dataframe(msg["content"])
            elif msg.get("type") == "markdown":
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])
                
    st.session_state.download_data = None # Reset pending downloads
    
    # Handle New Input from Chat
    if prompt := st.chat_input("Ask a question about your data..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "markdown"})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant processing
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                call = get_tool_call(prompt)
                tool = call.get('tool_name')
                args = call.get('arguments', {})

                if tool == "run_sql_query":
                    result = run_sql_query(**args)
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        st.session_state.messages.append({"role": "assistant", "content": result, "type": "dataframe"})
                    else:
                         st.markdown(result)
                         st.session_state.messages.append({"role": "assistant", "content": result, "type": "markdown"})
                         
                elif tool == "save_table_to_csv":
                    res_msg = save_table_to_csv(**args)
                    st.markdown(res_msg)
                    st.session_state.messages.append({"role": "assistant", "content": res_msg, "type": "markdown"})
                    if st.session_state.download_data:
                        st.download_button(
                            label="Download CSV",
                            data=st.session_state.download_data["data"],
                            file_name=st.session_state.download_data["filename"],
                            mime="text/csv",
                            key=str(len(st.session_state.messages)) # Ensure unique key for the button
                        )
                elif tool == "generate_plot":
                    res_msg = generate_plot(**args)
                    st.session_state.messages.append({"role": "assistant", "content": res_msg, "type": "markdown"})

