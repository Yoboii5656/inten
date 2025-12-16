import streamlit as st
import pandas as pd
import os
import json
import io
from datetime import datetime
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Natural Language SQL Query",
    page_icon="🔍",
    layout="wide"
)

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

def execute_sql(sql_query):
    engine = get_engine()
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
            explain_sql = f"EXPLAIN ANALYZE {sql_query}"
            result = conn.execute(text(explain_sql))
            rows = result.fetchall()
            plan_text = "\n".join([row[0] for row in rows])
            return plan_text, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_vanna():
    if not OPENAI_API_KEY:
        return None
    try:
        from vanna.openai import OpenAI_Chat
        from vanna.chromadb import ChromaDB_VectorStore
        
        class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
            def __init__(self, config=None):
                ChromaDB_VectorStore.__init__(self, config=config)
                OpenAI_Chat.__init__(self, config=config)
        
        vn = MyVanna(config={
            'api_key': OPENAI_API_KEY,
            'model': 'gpt-5'
        })
        
        vn.connect_to_postgres(
            host=os.environ.get("PGHOST"),
            dbname=os.environ.get("PGDATABASE"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD"),
            port=int(os.environ.get("PGPORT", 5432))
        )
        
        return vn
    except Exception as e:
        st.error(f"Error initializing Vanna: {e}")
        return None

def train_vanna_on_schema(vn):
    ddl_statements = '''
    CREATE TABLE workspaces (
        id UUID PRIMARY KEY,
        name TEXT NOT NULL,
        owner_id UUID,
        created_at TIMESTAMP,
        plan TEXT DEFAULT 'free',
        status TEXT DEFAULT 'active'
    );
    CREATE TABLE users (
        id UUID PRIMARY KEY,
        email TEXT NOT NULL,
        name TEXT NOT NULL,
        workspace_id UUID REFERENCES workspaces(id),
        role TEXT DEFAULT 'member',
        created_at TIMESTAMP
    );
    CREATE TABLE agents (
        id UUID PRIMARY KEY,
        workspace_id UUID REFERENCES workspaces(id),
        name TEXT NOT NULL,
        language TEXT DEFAULT 'en',
        llm_model TEXT DEFAULT 'gpt-4',
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP
    );
    CREATE TABLE integrations (
        id UUID PRIMARY KEY,
        workspace_id UUID REFERENCES workspaces(id),
        type TEXT NOT NULL,
        config JSONB,
        status TEXT DEFAULT 'active',
        last_sync_at TIMESTAMP,
        created_at TIMESTAMP
    );
    CREATE TABLE agent_tools (
        id UUID PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        tool_name TEXT NOT NULL,
        tool_config JSONB,
        created_at TIMESTAMP
    );
    CREATE TABLE agent_runs (
        id UUID PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        workspace_id UUID REFERENCES workspaces(id),
        run_type TEXT DEFAULT 'test_call',
        status TEXT DEFAULT 'success',
        duration_ms INTEGER,
        started_at TIMESTAMP,
        completed_at TIMESTAMP
    );
    CREATE TABLE test_runs (
        id UUID PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        workspace_id UUID REFERENCES workspaces(id),
        test_input TEXT,
        expected_output TEXT,
        actual_output TEXT,
        result TEXT DEFAULT 'pass',
        error_message TEXT,
        created_at TIMESTAMP
    );
    CREATE TABLE run_logs (
        id UUID PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id),
        step INTEGER,
        event_type TEXT,
        message TEXT,
        payload JSONB,
        timestamp TIMESTAMP
    );
    CREATE TABLE errors (
        id UUID PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id),
        workspace_id UUID REFERENCES workspaces(id),
        source TEXT,
        code TEXT,
        message TEXT,
        metadata JSONB,
        created_at TIMESTAMP
    );
    CREATE TABLE integration_sync_logs (
        id UUID PRIMARY KEY,
        integration_id UUID REFERENCES integrations(id),
        workspace_id UUID REFERENCES workspaces(id),
        sync_type TEXT,
        status TEXT DEFAULT 'success',
        items_synced INTEGER,
        error_message TEXT,
        created_at TIMESTAMP
    );
    CREATE TABLE billing_usage (
        id UUID PRIMARY KEY,
        workspace_id UUID REFERENCES workspaces(id),
        agent_id UUID REFERENCES agents(id),
        characters_generated INTEGER,
        calls_made INTEGER,
        tokens_used INTEGER,
        total_cost_usd DECIMAL(10,4),
        created_at TIMESTAMP
    );
    CREATE TABLE audit_events (
        id UUID PRIMARY KEY,
        workspace_id UUID REFERENCES workspaces(id),
        user_id UUID REFERENCES users(id),
        action TEXT,
        entity TEXT,
        before JSONB,
        after JSONB,
        created_at TIMESTAMP
    );
    '''
    
    documentation = '''
    This database stores data for a multi-tenant voice agent platform.
    Tables and their purposes:
    - workspaces: Customer accounts/organizations. Each workspace has a plan (free/pro/enterprise) and status (active/suspended).
    - users: Users belonging to workspaces with roles (admin/member/viewer).
    - agents: Voice agents configured in each workspace. Each agent has a language code (en=English, hi=Hindi, gu=Gujarati, ta=Tamil, te=Telugu, mr=Marathi, bn=Bengali, kn=Kannada, ml=Malayalam, pa=Punjabi) and an LLM model.
    - integrations: External integrations like HubSpot, Zoho, Twilio, Plivo, Slack, or custom APIs. Status can be active, inactive, or error.
    - agent_tools: Tools attached to agents like CRM Lookup, Email Sender, HubSpot API, Slack Notifier, Web Search, Database Query, Custom Script, Calendar Check.
    - agent_runs: Records of agent executions. run_type can be live_call, test_call, cron_job, or webhook. status is success, failed, or partial.
    - test_runs: Test executions with result being pass or fail.
    - run_logs: Detailed logs during agent runs with event_type being llm_call, tool_call, user_input, or system.
    - errors: System-level and run-level failures with source being integration, agent, llm, or tool.
    - integration_sync_logs: Sync activity logs for integrations like HubSpot/Zoho data fetching. sync_type is contacts, deals, or activities.
    - billing_usage: Usage metrics per agent including characters, calls, tokens, and cost.
    - audit_events: Change tracking for who did what, with action being create, update, or delete.
    '''
    
    sample_queries = [
        ("Show me all the failed test runs for agent X last week", 
         "SELECT tr.*, a.name as agent_name FROM test_runs tr JOIN agents a ON tr.agent_id = a.id WHERE tr.result = 'fail' AND tr.created_at >= NOW() - INTERVAL '7 days' ORDER BY tr.created_at DESC"),
        ("Which integrations are inactive for workspace Y",
         "SELECT i.*, w.name as workspace_name FROM integrations i JOIN workspaces w ON i.workspace_id = w.id WHERE i.status = 'inactive'"),
        ("Top 5 errors for the last 24 hours",
         "SELECT code, message, source, COUNT(*) as error_count FROM errors WHERE created_at >= NOW() - INTERVAL '24 hours' GROUP BY code, message, source ORDER BY error_count DESC LIMIT 5"),
        ("List all agents using Gujarati",
         "SELECT a.*, w.name as workspace_name FROM agents a JOIN workspaces w ON a.workspace_id = w.id WHERE a.language = 'gu'"),
        ("Show all sync logs for HubSpot integrations",
         "SELECT isl.*, i.type as integration_type, w.name as workspace_name FROM integration_sync_logs isl JOIN integrations i ON isl.integration_id = i.id JOIN workspaces w ON isl.workspace_id = w.id WHERE i.type = 'hubspot' ORDER BY isl.created_at DESC"),
        ("What tool calls caused the most failures",
         "SELECT rl.message, COUNT(*) as failure_count FROM run_logs rl JOIN agent_runs ar ON rl.run_id = ar.id WHERE rl.event_type = 'tool_call' AND ar.status = 'failed' GROUP BY rl.message ORDER BY failure_count DESC LIMIT 10"),
        ("Show all workspaces",
         "SELECT * FROM workspaces ORDER BY created_at DESC"),
        ("Count agents per workspace",
         "SELECT w.name as workspace_name, COUNT(a.id) as agent_count FROM workspaces w LEFT JOIN agents a ON w.id = a.workspace_id GROUP BY w.id, w.name ORDER BY agent_count DESC"),
    ]
    
    try:
        vn.train(ddl=ddl_statements)
        vn.train(documentation=documentation)
        for question, sql in sample_queries:
            vn.train(question=question, sql=sql)
        return True
    except Exception as e:
        st.error(f"Error training Vanna: {e}")
        return False

def get_db_stats():
    engine = get_engine()
    stats = {}
    tables = ['workspaces', 'users', 'agents', 'integrations', 'agent_tools', 
              'agent_runs', 'test_runs', 'run_logs', 'errors', 
              'integration_sync_logs', 'billing_usage', 'audit_events']
    
    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            stats[table] = result.scalar()
    return stats

def get_workspaces():
    df, error = execute_sql("SELECT id, name FROM workspaces ORDER BY name")
    if error:
        return []
    return [(row['id'], row['name']) for _, row in df.iterrows()]

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
        "Top Errors (Last 24h)": "SELECT code, message, source, COUNT(*) as error_count FROM errors WHERE created_at >= NOW() - INTERVAL '24 hours' GROUP BY code, message, source ORDER BY error_count DESC LIMIT 10",
        "Errors by Source": "SELECT source, COUNT(*) as count FROM errors GROUP BY source ORDER BY count DESC",
        "Error Trend (Last 7 Days)": "SELECT DATE(created_at) as date, COUNT(*) as error_count FROM errors WHERE created_at >= NOW() - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY date",
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
        "Daily Usage (Last 7 Days)": "SELECT DATE(created_at) as date, SUM(tokens_used) as tokens, SUM(calls_made) as calls FROM billing_usage WHERE created_at >= NOW() - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY date",
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
    
    st.divider()
    
    st.header("🏢 Workspace Filter")
    workspaces = get_workspaces()
    workspace_options = ["All Workspaces"] + [w[1] for w in workspaces]
    selected_workspace = st.selectbox("Filter by Workspace", workspace_options)
    
    if selected_workspace != "All Workspaces":
        workspace_id = next((w[0] for w in workspaces if w[1] == selected_workspace), None)
        st.session_state.workspace_filter = workspace_id
    else:
        st.session_state.workspace_filter = None
    
    st.divider()
    
    st.header("📝 Query Templates")
    template_category = st.selectbox("Category", list(QUERY_TEMPLATES.keys()))
    template_queries = QUERY_TEMPLATES[template_category]
    selected_template = st.selectbox("Select Template", list(template_queries.keys()))
    
    if st.button("Use Template", key="use_template"):
        st.session_state.template_sql = template_queries[selected_template]

tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Analytics Dashboard", "💾 Saved Queries"])

with tab1:
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API key is required for natural language queries. Add the OPENAI_API_KEY secret.")
        st.info("You can still use SQL templates and raw SQL queries below.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if hasattr(st.session_state, 'template_sql') and st.session_state.template_sql:
            st.subheader("Template SQL")
            sql_to_run = st.text_area("SQL Query", value=st.session_state.template_sql, height=100)
            use_nl = False
        else:
            if OPENAI_API_KEY:
                question = st.text_input(
                    "Ask a question about your data:",
                    placeholder="e.g., Show me all failed test runs from last week"
                )
                use_nl = True
            else:
                sql_to_run = st.text_area("Enter SQL Query:", height=100, 
                                          placeholder="SELECT * FROM workspaces LIMIT 10")
                use_nl = False
    
    with col2:
        show_sql = st.checkbox("Show SQL", value=True)
        show_chart = st.checkbox("Auto-generate Chart", value=True)
        show_explain = st.checkbox("Show Query Plan", value=False)
    
    if st.button("🚀 Run Query", type="primary"):
        if use_nl and OPENAI_API_KEY:
            vn = get_vanna()
            if vn and not st.session_state.vanna_trained:
                with st.spinner("Training AI on database schema..."):
                    success = train_vanna_on_schema(vn)
                    if success:
                        st.session_state.vanna_trained = True
            
            if vn and question:
                with st.spinner("Generating SQL..."):
                    try:
                        workspace_hint = ""
                        if st.session_state.get('workspace_filter'):
                            workspace_hint = f" (filter by workspace_id = '{st.session_state.workspace_filter}')"
                        
                        sql = vn.generate_sql(question + workspace_hint)
                        st.session_state.last_sql = sql
                        
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
        error_trend_sql = "SELECT DATE(created_at) as date, COUNT(*) as error_count FROM errors WHERE created_at >= NOW() - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY date"
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
    usage_sql = "SELECT DATE(created_at) as date, SUM(tokens_used) as tokens, SUM(calls_made) as calls FROM billing_usage WHERE created_at >= NOW() - INTERVAL '30 days' GROUP BY DATE(created_at) ORDER BY date"
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
st.caption("Powered by Vanna.ai - Natural Language to SQL")
