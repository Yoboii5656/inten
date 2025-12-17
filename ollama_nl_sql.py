"""
Ollama Local LLM powered Natural Language to SQL Converter
Runs completely locally - no API costs, no internet needed, complete privacy
"""

import os
import requests
from typing import Tuple, Optional

class OllamaNLtoSQL:
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama NL-to-SQL converter
        
        Args:
            model: Ollama model to use (e.g., 'llama3.1', 'mistral', 'codellama', 'deepseek-coder')
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        if not self._test_connection():
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Make sure Ollama is running. Install from: https://ollama.ai"
            )
        
        self.schema_info = self._get_default_schema()
    
    def _test_connection(self) -> bool:
        """Test if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_default_schema(self) -> str:
        """Get concise database schema with EXACT column names"""
        return """
DATABASE: Agent Platform (SQLite)

IMPORTANT: Use EXACT column names shown below!

TABLES WITH EXACT COLUMNS:

workspaces:
  id, name, owner_id, created_at, plan, status

users:
  id, email, name, workspace_id, role, created_at

agents:
  id, workspace_id, name, language, llm_model, status, created_at

integrations:
  id, workspace_id, type, config, status, last_sync_at, created_at

agent_tools:
  id, agent_id, tool_name, tool_config, created_at

agent_runs:
  id, agent_id, workspace_id, run_type, status, duration_ms, started_at, completed_at

test_runs:
  id, agent_id, workspace_id, test_input, expected_output, actual_output, result, error_message, created_at

run_logs:
  id, run_id, step, event_type, message, payload, timestamp

errors:
  id, run_id, workspace_id, source, code, message, metadata, created_at

integration_sync_logs:
  id, integration_id, workspace_id, sync_type, status, items_synced, error_message, created_at

billing_usage:
  id, workspace_id, agent_id, characters_generated, calls_made, tokens_used, total_cost_usd, created_at

audit_events:
  id, workspace_id, user_id, action, entity, before, after, created_at

COLUMN VALUES:
- plan: 'free', 'pro', 'enterprise'
- status: 'active', 'suspended', 'inactive'
- role: 'admin', 'member', 'viewer'
- language: 'en', 'hi', 'gu', 'ta', 'te', 'mr', 'bn', 'kn', 'ml', 'pa'
- result: 'pass', 'fail'
- run_type: 'live_call', 'test_call', 'cron_job', 'webhook'
- event_type: 'llm_call', 'tool_call', 'user_input', 'system'
- source: 'integration', 'agent', 'llm', 'tool'

RELATIONSHIPS:
- workspaces.id → users.workspace_id, agents.workspace_id, integrations.workspace_id
- agents.id → agent_runs.agent_id, test_runs.agent_id, agent_tools.agent_id
- agent_runs.id → run_logs.run_id, errors.run_id

TIME FILTERS (SQLite):
- Last 24h: WHERE created_at >= datetime('now', '-24 hours')
- Last 7d: WHERE created_at >= datetime('now', '-7 days')
- Today: WHERE date(created_at) = date('now')

EXAMPLES:
Q: "Top 5 errors last 24h"
A: SELECT code, message, COUNT(*) as cnt FROM errors WHERE created_at >= datetime('now', '-24 hours') GROUP BY code, message ORDER BY cnt DESC LIMIT 5

Q: "All Hindi agents"
A: SELECT name FROM agents WHERE language='hi' AND status='active'

Q: "Failed runs today"
A: SELECT * FROM agent_runs WHERE status='failed' AND date(started_at)=date('now') ORDER BY started_at DESC
"""
    
    def parse_question(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert natural language question to SQL query using Ollama
        
        Args:
            question: Natural language question
            
        Returns:
            Tuple of (sql_query, explanation)
        """
        try:
            prompt = f"""Convert this question to SQLite query.

{self.schema_info}

RULES:
- Return ONLY SQL query, no explanations
- Use SQLite syntax with datetime() for time filters
- Use JOINs for related tables
- SELECT only (read-only)
- Add ORDER BY and LIMIT

QUESTION: {question}

SQL:"""

            # Call Ollama API
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent SQL
                        "top_p": 0.9,
                    }
                },
                timeout=100  # 1 minutes timeout for slower models
            )
            
            if response.status_code != 200:
                return None, f"Ollama API error: {response.status_code}"
            
            result = response.json()
            sql = result.get('response', '').strip()
            
            # Clean up the SQL
            sql = self._clean_sql(sql)
            
            if not sql:
                return None, "Could not generate SQL query. Please try rephrasing your question."
            
            # Validate it's a SELECT query
            if not sql.upper().startswith('SELECT'):
                return None, "Only SELECT queries are allowed for safety."
            
            explanation = f"Generated SQL using Local AI ({self.model})"
            return sql, explanation
            
        except requests.exceptions.Timeout:
            return None, "Local AI request timed out. The model might be slow or not loaded."
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to Local AI. Make sure Ollama service is running."
        except Exception as e:
            return None, f"Error generating SQL: {str(e)}"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up SQL query from LLM response"""
        # Remove markdown code blocks
        sql = sql.replace('```sql', '').replace('```', '')
        
        # Remove common prefixes
        prefixes = ['SQL:', 'Query:', 'Answer:', 'SELECT']
        for prefix in prefixes[:-1]:  # Don't remove SELECT
            if sql.startswith(prefix):
                sql = sql[len(prefix):].strip()
        
        # Take only the first query if multiple
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            if line.upper().startswith('SELECT'):
                sql_lines = [line]
            elif sql_lines:
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        sql = ' '.join(sql_lines) if sql_lines else sql.split('\n')[0]
        
        # Remove trailing semicolon
        sql = sql.rstrip(';').strip()
        
        return sql
    
    def get_suggestions(self) -> list:
        """Get list of example questions based on actual database content"""
        return [
            # Workspace queries
            "Show all workspaces and their plans",
            "Which workspaces are on enterprise plan?",
            "List suspended workspaces",
            "Count agents per workspace",
            
            # Agent queries
            "Show all agents using Hindi language",
            "List all Gujarati and Tamil agents",
            "Show inactive or draft agents",
            "How many agents use each language?",
            
            # Error and failure analysis
            "Top 5 error codes from last week",
            "Show all errors from integration source",
            
            
            # Test and run analysis
            "Show all failed test runs from last month",
            "Which runs took longer than 5 seconds?",
            
            # Integration queries
            "Which integrations failed to sync?",
            "List inactive integrations",
            
            # Billing and usage
            "Total billing cost by workspace",
            "Show top 5 workspaces by total cost",
            "Billing usage for last 30 days",
            
            # User queries
            "How many admin users per workspace?",
            
            # Complex queries
            "Most common error messages",
        ]
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []
    
    def test_connection(self) -> bool:
        """Test if Ollama is working"""
        return self._test_connection()
