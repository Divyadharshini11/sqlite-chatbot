import sqlite3
import re
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process
import streamlit as st

# ============================================================
# STREAMLIT SETUP
# ============================================================
st.set_page_config(page_title="Data Chatbot", page_icon="🤖", layout="wide")
st.title("SQLite Database Chatbot")

# ============================================================
# SESSION STATE
# ============================================================
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
# DATABASE UPLOAD
# ============================================================
with st.sidebar:
    st.header("Upload Database")
    uploaded_file = st.file_uploader("Upload SQLite DB", type=["sqlite", "db", "sqlite3"])

    if uploaded_file and st.session_state.db_conn is None:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        conn = sqlite3.connect(temp_path, check_same_thread=False)
        st.session_state.db_conn = conn
        cursor = conn.cursor()

        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        st.session_state.tables = tables

        # Get schema
        schema = ""
        column_names = []

        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = cursor.fetchall()
            schema += f"\nTable: {table}\nColumns: " + ", ".join([c[1] for c in cols])
            column_names.extend([c[1] for c in cols])

        st.session_state.schema = schema
        st.session_state.column_names = column_names

        st.success(f"Loaded tables: {', '.join(tables)}")

# ============================================================
# CORE FUNCTIONS
# ============================================================
def run_sql_query(sql_query):
    try:
        sql_query = sql_query.split(";")[0]
        st.markdown(f"`Running SQL: {sql_query}`")
        return pd.read_sql_query(sql_query, st.session_state.db_conn)
    except Exception as e:
        return f"Error: {e}"


def generate_plot(table, x_col, y_col, plot_type="line"):
    try:
        df = pd.read_sql_query(
            f'SELECT "{x_col}", "{y_col}" FROM "{table}" LIMIT 500',
            st.session_state.db_conn,
        )

        fig, ax = plt.subplots()

        if plot_type == "line":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        else:
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)

        st.pyplot(fig)
        return "Plot created"
    except Exception as e:
        return f"Plot Error: {e}"


def save_table_to_csv(table_name):
    best = process.extractOne(table_name, st.session_state.tables, score_cutoff=60)

    if not best:
        return "Table not found"

    table = best[0]
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', st.session_state.db_conn)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{table}.csv", "text/csv")

    return "Download ready"


# ============================================================
# QUERY ROUTER (NO LLM)
# ============================================================
def get_tool_call(question):
    q = question.lower()
    cols = st.session_state.column_names
    tables = st.session_state.tables

    # Aggregation
    agg_map = {
        "average": "AVG",
        "mean": "AVG",
        "max": "MAX",
        "min": "MIN",
        "sum": "SUM",
        "count": "COUNT",
    }

    for word, func in agg_map.items():
        if word in q:
            col = next((c for c in cols if c.lower() in q), None)
            tab = process.extractOne(q, tables, score_cutoff=60)

            if col and tab:
                sql = f'SELECT {func}("{col}") FROM "{tab[0]}"'
                return "sql", sql

    # Plot
    if any(w in q for w in ["plot", "chart", "graph"]):
        tab = process.extractOne(q, tables, score_cutoff=60)
        mentioned = [c for c in cols if c.lower() in q]

        if tab and len(mentioned) >= 2:
            return "plot", (tab[0], mentioned[0], mentioned[1])

    # Export
    if any(w in q for w in ["download", "export", "csv"]):
        tab = process.extractOne(q, tables, score_cutoff=60)
        if tab:
            return "export", tab[0]

    # Default fallback
    if tables:
        return "sql", f'SELECT * FROM "{tables[0]}" LIMIT 10'

    return "none", None


# ============================================================
# CHAT UI
# ============================================================
if st.session_state.db_conn is None:
    st.info("Upload a database to start")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "df":
                st.dataframe(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            tool, data = get_tool_call(prompt)

            if tool == "sql":
                result = run_sql_query(data)

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result, "type": "df"}
                    )
                else:
                    st.markdown(result)

            elif tool == "plot":
                msg = generate_plot(*data)
                st.markdown(msg)

            elif tool == "export":
                msg = save_table_to_csv(data)
                st.markdown(msg)

            else:
                st.markdown("Could not understand request")
