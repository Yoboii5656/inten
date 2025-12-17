# Agent Platform Analytics – Natural Language SQL Interface

A Streamlit-based analytics dashboard that enables users to query an Agent Platform database using natural language. The system converts plain English questions into SQL using a **local large language model (Ollama)** and executes them safely against a SQLite database.

All processing is performed locally. No external APIs or cloud services are used.

---

## Overview

This project provides a natural language interface for analyzing operational, billing, and performance data of a multi-tenant agent platform. It is designed for internal analytics, debugging, and business intelligence use cases.

The application:
- Translates natural language questions into SQL
- Executes read-only queries securely
- Displays results in tabular and visual formats
- Runs entirely on local infrastructure

---

## Features

### Natural Language to SQL
- Plain English query input
- Automatic SQL generation using a local LLM (Ollama)
- Supports joins, filters, grouping, and aggregations
- Enforces SELECT-only queries for safety

### Data Visualization
- Automatic chart generation based on query results
- Interactive visualizations using Plotly
- Supports bar, line, pie, and scatter charts

### Analytics & Debugging
- SQL preview with syntax highlighting
- Query execution metrics
- EXPLAIN plan support
- Export results to CSV, Excel, or JSON

### Privacy & Security
- Fully local execution
- No internet dependency after setup
- No third-party API calls
- SQL injection prevention via SQLAlchemy

---

## Tech Stack

**Frontend**
- Streamlit
- Plotly
- Pandas

**Backend**
- Python 3.12+
- SQLAlchemy
- SQLite

**AI**
- Ollama (local LLM runtime)
- Mistral (default NL-to-SQL model)

---

## Prerequisites

- Python 3.12+
- Ollama installed and running
- 8 GB RAM minimum (16 GB recommended)
- Windows, macOS, or Linux

---

## Installation & Setup

### 1. Install Ollama and Model
```bash
ollama pull mistral
