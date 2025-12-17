import streamlit as st
import pandas as pd
import os
import json
import io
from datetime import datetime
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


st.set_page_config(
    page_title="Natural Language SQL Query",
    page_icon="🔍",
    layout="wide"
)

from schema_data import DDL_STATEMENTS, DOCUMENTATION, SAMPLE_QUERIES
from ollama_nl_sql import OllamaNLtoSQL
from local_nl_sql import LocalNLtoSQL  # Fallback only
# Removed: Gemini, OpenAI - Using Ollama only

# Initialize session state configuration
if 'db_url' not in st.session_state:
    st.session_state.db_url = os.environ.get("DATABASE_URL", "")
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
if 'nl_parser' not in st.session_state:
    # Use ONLY Ollama, fallback to local patterns if Ollama unavailable
    try:
        st.session_state.nl_parser = OllamaNLtoSQL(model=st.session_state.ollama_model)
        st.session_state.parser_type = "ollama"
    except Exception as e:
        print(f"Ollama not available: {e}")
        st.session_state.nl_parser = LocalNLtoSQL()
        st.session_state.parser_type = "local"

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Configuration form
    with st.expander("Connection Settings", expanded=not st.session_state.db_url):
        new_db_url = st.text_input(
            "Database URL", 
            value=st.session_state.db_url,
            type="password",
            help="sqlite:///path/to/database.db"
        )
        
        st.markdown("### 🏠 Local AI Model")
        st.info("Using local AI model for completely private and free NL-to-SQL conversion.")
        
        new_ollama_model = st.text_input(
            "AI Model",
            value=st.session_state.ollama_model,
            help="Recommended: llama3.1, mistral, codellama, deepseek-coder"
        )
        
        st.markdown("### 📝 Quick Start")
        st.code("ollama pull llama3.1", language="bash")
        st.caption("Run this command to install the model")
        
        if st.button("Save Configuration"):
            st.session_state.db_url = new_db_url
            st.session_state.ollama_model = new_ollama_model
            
            # Reinitialize with Ollama only
            try:
                st.session_state.nl_parser = OllamaNLtoSQL(model=new_ollama_model)
                st.session_state.parser_type = "ollama"
                st.success(f"✅ Using Local AI ({new_ollama_model})")
            except Exception as e:
                st.session_state.nl_parser = LocalNLtoSQL()
                st.session_state.parser_type = "local"
                st.error(f"❌ Local AI not available. Install Ollama from: https://ollama.ai")
                st.error(f"Error: {str(e)[:200]}")
            
            st.rerun()

    if not st.session_state.db_url:
        st.error("Please configure Database URL to proceed")
        st.stop()

# Use session state values
DATABASE_URL = st.session_state.db_url
nl_parser = st.session_state.nl_parser

@st.cache_resource
def get_engine():
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        return None

def execute_sql(sql_query):
    engine = get_engine()
    if engine is None:
        return None, "Invalid Database Configuration"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df, None
    except Exception as e:
        return None, str(e)

def get_explain_plan(sql_query):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            # SQLite uses EXPLAIN QUERY PLAN instead of EXPLAIN ANALYZE
            if 'sqlite' in DATABASE_URL.lower():
                explain_sql = f"EXPLAIN QUERY PLAN {sql_query}"
            else:
                explain_sql = f"EXPLAIN ANALYZE {sql_query}"
            result = conn.execute(text(explain_sql))
            rows = result.fetchall()
            plan_text = "\n".join([str(row[0]) if len(row) == 1 else ' | '.join(map(str, row)) for row in rows])
            return plan_text, None
    except Exception as e:
        return None, str(e)

# Local NL-to-SQL parser is initialized in session state
# No external API calls needed!

def get_db_stats():
    engine = get_engine()
    stats = {}
    tables = ['workspaces', 'users', 'agents', 'integrations', 'agent_tools', 
              'agent_runs', 'test_runs', 'run_logs', 'errors', 
              'integration_sync_logs', 'billing_usage', 'audit_events']
    
    try:
        with engine.connect() as conn:
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[table] = result.scalar()
                except Exception:
                    stats[table] = 0
    except Exception as e:
        st.sidebar.error(f"Connection Error: {str(e)}")
        return {}
        
    return stats

def get_workspaces():
    try:
        df, error = execute_sql("SELECT id, name FROM workspaces ORDER BY name")
        if error:
            return []
        return [(row['id'], row['name']) for _, row in df.iterrows()]
    except Exception:
        return []

def export_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def export_to_json(df):
    return df.to_json(orient='records', indent=2).encode('utf-8')

def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Query Results')
    return output.getvalue()

QUERY_TEMPLATES = {
    "Error Analysis": {
        "Top Errors (Last 24h)": "SELECT code, message, source, COUNT(*) as error_count FROM errors WHERE created_at >= datetime('now', '-24 hours') GROUP BY code, message, source ORDER BY error_count DESC LIMIT 10",
        "Errors by Source": "SELECT source, COUNT(*) as count FROM errors GROUP BY source ORDER BY count DESC",
        "Error Trend (Last 7 Days)": "SELECT date(created_at) as date, COUNT(*) as error_count FROM errors WHERE created_at >= datetime('now', '-7 days') GROUP BY date(created_at) ORDER BY date",
    },
    "Agent Performance": {
        "Agent Run Status Summary": "SELECT a.name as agent_name, ar.status, COUNT(*) as count FROM agent_runs ar JOIN agents a ON ar.agent_id = a.id GROUP BY a.name, ar.status ORDER BY a.name",
        "Test Pass/Fail Rates": "SELECT a.name as agent_name, tr.result, COUNT(*) as count FROM test_runs tr JOIN agents a ON tr.agent_id = a.id GROUP BY a.name, tr.result ORDER BY a.name",
        "Avg Run Duration by Agent": "SELECT a.name as agent_name, AVG(ar.duration_ms) as avg_duration_ms FROM agent_runs ar JOIN agents a ON ar.agent_id = a.id GROUP BY a.name ORDER BY avg_duration_ms DESC",
    },
    "Integration Health": {
        "Integration Status Overview": "SELECT type, status, COUNT(*) as count FROM integrations GROUP BY type, status ORDER BY type",
        "Inactive Integrations": "SELECT i.*, w.name as workspace_name FROM integrations i JOIN workspaces w ON i.workspace_id = w.id WHERE i.status = 'inactive'",
        "Sync Success Rate": "SELECT i.type, isl.status, COUNT(*) as count FROM integration_sync_logs isl JOIN integrations i ON isl.integration_id = i.id GROUP BY i.type, isl.status ORDER BY i.type",
    },
    "Workspace Analytics": {
        "Workspaces by Plan": "SELECT plan, COUNT(*) as count FROM workspaces GROUP BY plan",
        "Agents per Workspace": "SELECT w.name as workspace_name, COUNT(a.id) as agent_count FROM workspaces w LEFT JOIN agents a ON w.id = a.workspace_id GROUP BY w.id, w.name ORDER BY agent_count DESC",
        "Billing Summary by Workspace": "SELECT w.name as workspace_name, SUM(b.total_cost_usd) as total_cost, SUM(b.tokens_used) as total_tokens FROM billing_usage b JOIN workspaces w ON b.workspace_id = w.id GROUP BY w.id, w.name ORDER BY total_cost DESC",
    },
    "Usage Metrics": {
        "Daily Usage (Last 7 Days)": "SELECT date(created_at) as date, SUM(tokens_used) as tokens, SUM(calls_made) as calls FROM billing_usage WHERE created_at >= datetime('now', '-7 days') GROUP BY date(created_at) ORDER BY date",
        "Top Token Users": "SELECT a.name as agent_name, SUM(b.tokens_used) as total_tokens FROM billing_usage b JOIN agents a ON b.agent_id = a.id GROUP BY a.name ORDER BY total_tokens DESC LIMIT 10",
    }
}

def detect_chart_type(df):
    if len(df.columns) < 2:
        return None
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'created_at' in col.lower() or 'timestamp' in col.lower()]
    
    if date_cols and numeric_cols:
        return 'line'
    elif len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
        if len(df) <= 20:
            return 'bar'
        else:
            return 'bar'
    
    return None

def create_chart(df, chart_type):
    if chart_type is None or len(df) == 0:
        return None
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if len(numeric_cols) == 0 or len(non_numeric_cols) == 0:
        return None
    
    x_col = non_numeric_cols[0]
    y_col = numeric_cols[0]
    
    if chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
    elif chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    elif chart_type == 'pie':
        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} Distribution")
    else:
        return None
    
    fig.update_layout(height=400)
    return fig

st.title("🔍 Natural Language SQL Query System")
st.markdown("Ask questions about your agent platform data in plain English")

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "vanna_trained" not in st.session_state:
    st.session_state.vanna_trained = False
if "saved_queries" not in st.session_state:
    st.session_state.saved_queries = []
if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None

with st.sidebar:
    st.header("📊 Database Statistics")
    if st.button("Refresh Stats"):
        st.cache_data.clear()
    
    try:
        stats = get_db_stats()
        cols = st.columns(2)
        for i, (table, count) in enumerate(stats.items()):
            with cols[i % 2]:
                st.metric(table.replace("_", " ").title(), count)
    except Exception as e:
        st.error(f"Error loading stats: {e}")

tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Analytics Dashboard", "💾 Saved Queries"])

with tab1:
    # Show parser type and example questions
    parser_type = st.session_state.get('parser_type', 'local')
    if parser_type == 'ollama':
        st.success("🏠 **Using Local AI Model** - 100% Private & Free!")
    else:
        st.warning("⚠️ **Local AI not available** - Using basic pattern matching. Install Ollama from: https://ollama.ai")
    
    with st.expander("📝 Example Questions You Can Ask"):
        suggestions = nl_parser.get_suggestions()
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                st.markdown(f"- {suggestion}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., Show me all workspaces with their agent counts and total billing"
        )
        use_nl = True
    
    with col2:
        show_sql = st.checkbox("Show SQL", value=True)
        show_chart = st.checkbox("Auto-generate Chart", value=True)
        show_explain = st.checkbox("Show Query Plan", value=False)
    
    if st.button("🚀 Run Query", type="primary"):
        if use_nl and question:
            # Use local NL-to-SQL parser
            with st.spinner("Parsing your question..."):
                try:
                    sql, explanation = nl_parser.parse_question(question)
                    
                    if sql is None:
                        st.error(f"❌ {explanation}")
                        st.info("Try rephrasing your question or use one of the example questions above.")
                    else:
                        st.session_state.last_sql = sql
                        
                        if explanation:
                            st.success(f"✅ {explanation}")
                        
                        if show_sql:
                            st.subheader("Generated SQL")
                            st.code(sql, language="sql")
                        
                        start_time = datetime.now()
                        df, error = execute_sql(sql)
                        execution_time = (datetime.now() - start_time).total_seconds()
                        
                        if error:
                            st.error(f"Query execution error: {error}")
                        else:
                            st.success(f"Query returned {len(df)} rows in {execution_time:.3f} seconds")
                            st.session_state.last_result_df = df
                            st.dataframe(df, use_container_width=True)
                            
                            if show_chart and len(df) > 0:
                                chart_type = detect_chart_type(df)
                                if chart_type:
                                    fig = create_chart(df, chart_type)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            if show_explain:
                                plan, plan_error = get_explain_plan(sql)
                                if plan:
                                    st.subheader("Query Execution Plan")
                                    st.code(plan, language="text")
                                    
                                    if "Seq Scan" in plan:
                                        st.warning("⚠️ Sequential scan detected. Consider adding an index for better performance.")
                                    if "Index Scan" in plan or "Index Only Scan" in plan:
                                        st.success("✅ Query is using indexes efficiently.")
                            
                            st.session_state.query_history.append({
                                "question": question,
                                "sql": sql,
                                "rows": len(df),
                                "time": execution_time,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            sql = sql_to_run if 'sql_to_run' in dir() else st.session_state.get('template_sql', '')
            if sql:
                st.session_state.last_sql = sql
                
                if show_sql:
                    st.subheader("SQL Query")
                    st.code(sql, language="sql")
                
                start_time = datetime.now()
                df, error = execute_sql(sql)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if error:
                    st.error(f"Query execution error: {error}")
                else:
                    st.success(f"Query returned {len(df)} rows in {execution_time:.3f} seconds")
                    st.session_state.last_result_df = df
                    st.dataframe(df, use_container_width=True)
                    
                    if show_chart and len(df) > 0:
                        chart_type = detect_chart_type(df)
                        if chart_type:
                            fig = create_chart(df, chart_type)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    if show_explain:
                        plan, plan_error = get_explain_plan(sql)
                        if plan:
                            st.subheader("Query Execution Plan")
                            st.code(plan, language="text")
                            
                            if "Seq Scan" in plan:
                                st.warning("⚠️ Sequential scan detected. Consider adding an index for better performance.")
                            if "Index Scan" in plan or "Index Only Scan" in plan:
                                st.success("✅ Query is using indexes efficiently.")
        
        if hasattr(st.session_state, 'template_sql'):
            st.session_state.template_sql = None
    
    if st.session_state.last_result_df is not None and len(st.session_state.last_result_df) > 0:
        st.divider()
        st.subheader("📥 Export Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        df = st.session_state.last_result_df
        
        with col1:
            csv_data = export_to_csv(df)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = export_to_json(df)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            excel_data = export_to_excel(df)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col4:
            if st.session_state.last_sql:
                save_name = st.text_input("Query Name", placeholder="My Query")
                if st.button("Save Query"):
                    if save_name:
                        st.session_state.saved_queries.append({
                            "name": save_name,
                            "sql": st.session_state.last_sql,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                        st.success(f"Query '{save_name}' saved!")
    
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Query History")
        for entry in reversed(st.session_state.query_history[-5:]):
            with st.expander(f"[{entry['timestamp']}] {entry.get('question', 'SQL Query')[:50]}..."):
                st.write(f"**Question:** {entry.get('question', 'Direct SQL')}")
                st.code(entry['sql'], language="sql")
                st.write(f"**Results:** {entry['rows']} rows in {entry['time']:.3f}s")

with tab2:
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Error Trends (Last 7 Days)")
        error_trend_sql = "SELECT date(created_at) as date, COUNT(*) as error_count FROM errors WHERE created_at >= datetime('now', '-7 days') GROUP BY date(created_at) ORDER BY date"
        df, error = execute_sql(error_trend_sql)
        if not error and len(df) > 0:
            fig = px.line(df, x='date', y='error_count', title='Errors Over Time')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error data for the last 7 days")
    
    with col2:
        st.markdown("#### Errors by Source")
        error_source_sql = "SELECT source, COUNT(*) as count FROM errors GROUP BY source ORDER BY count DESC"
        df, error = execute_sql(error_source_sql)
        if not error and len(df) > 0:
            fig = px.pie(df, names='source', values='count', title='Error Distribution by Source')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error data available")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Agent Run Status")
        run_status_sql = "SELECT status, COUNT(*) as count FROM agent_runs GROUP BY status"
        df, error = execute_sql(run_status_sql)
        if not error and len(df) > 0:
            fig = px.bar(df, x='status', y='count', title='Agent Runs by Status', color='status')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent run data available")
    
    with col4:
        st.markdown("#### Integration Status")
        int_status_sql = "SELECT type, status, COUNT(*) as count FROM integrations GROUP BY type, status ORDER BY type"
        df, error = execute_sql(int_status_sql)
        if not error and len(df) > 0:
            fig = px.bar(df, x='type', y='count', color='status', title='Integrations by Type and Status', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No integration data available")
    
    st.markdown("#### Usage Over Time")
    usage_sql = "SELECT date(created_at) as date, SUM(tokens_used) as tokens, SUM(calls_made) as calls FROM billing_usage WHERE created_at >= datetime('now', '-30 days') GROUP BY date(created_at) ORDER BY date"
    df, error = execute_sql(usage_sql)
    if not error and len(df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['tokens'], name='Tokens Used', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df['date'], y=df['calls'], name='Calls Made', mode='lines+markers', yaxis='y2'))
        fig.update_layout(
            title='Usage Metrics Over Time',
            yaxis=dict(title='Tokens'),
            yaxis2=dict(title='Calls', overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No usage data available")
    
    st.markdown("#### Workspace Comparison")
    workspace_comparison_sql = """
    SELECT w.name as workspace,
           COUNT(DISTINCT a.id) as agents,
           COUNT(DISTINCT i.id) as integrations,
           COALESCE(SUM(b.total_cost_usd), 0) as total_cost
    FROM workspaces w
    LEFT JOIN agents a ON w.id = a.workspace_id
    LEFT JOIN integrations i ON w.id = i.workspace_id
    LEFT JOIN billing_usage b ON w.id = b.workspace_id
    GROUP BY w.id, w.name
    ORDER BY total_cost DESC
    LIMIT 10
    """
    df, error = execute_sql(workspace_comparison_sql)
    if not error and len(df) > 0:
        st.dataframe(df, use_container_width=True)
        
        fig = px.bar(df, x='workspace', y=['agents', 'integrations'], 
                     title='Agents and Integrations per Workspace',
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No workspace comparison data available")

with tab3:
    st.subheader("💾 Saved Queries")
    
    if st.session_state.saved_queries:
        for i, query in enumerate(st.session_state.saved_queries):
            with st.expander(f"{query['name']} - {query['timestamp']}"):
                st.code(query['sql'], language="sql")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run", key=f"run_{i}"):
                        df, error = execute_sql(query['sql'])
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.success(f"Query returned {len(df)} rows")
                            st.dataframe(df, use_container_width=True)
                with col2:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.saved_queries.pop(i)
                        st.rerun()
    else:
        st.info("No saved queries yet. Run a query and click 'Save Query' to save it.")

st.divider()
parser_type = st.session_state.get('parser_type', 'local')
if parser_type == 'ollama':
    st.caption("🏠 Powered by Local AI - Running on Your Machine | 100% Private & Free")
else:
    st.caption("⚠️ Pattern Matching Mode - Install Ollama for Local AI: https://ollama.ai")
