import openai
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from utils import extract_sql_from_response
import difflib
from datetime import datetime
import json
import re  # Added for parsing LLM output

class SQLPlanGeneratorAgent:
    def __init__(self, api_key, base_url, model="gpt-3.5-turbo", style="default"):
        self.api_key=api_key
        self.base_url=base_url
        self.client = None
        self.model_name = model
        self.style = style
    
    async def __aenter__(self):
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    async def cleanup(self):
        """Close the async client properly"""
        if self.client:
            await self.client.close()

    def get_latest_reflection(self):
        """Read the most recent reflection file from the knowledge directory."""
        knowledge_dir = "./knowledge"
        if not os.path.exists(knowledge_dir):
            return ""
        files = [f for f in os.listdir(knowledge_dir) if f.startswith("reflection_") and f.endswith(".txt")]
        if not files:
            return ""
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(knowledge_dir, x)))
        with open(os.path.join(knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            return f.read()

    async def generate_full_sql_async(self, question, schema_info):
        # ① Read latest reflection
        reflection = self.get_latest_reflection()

        # ② Plan 생성
        plan_prompt = f"""You are a planner. Given the question, schema, and previous reflections, decompose into SQL plan steps. DO NOT use user functions and REGEXP, GLOB, CAST.

# Question:
{question}

# Schema Info:
{schema_info}

# Previous Reflections:
{reflection}

# Give high-level steps to generate the SQL with smart and concise."""
        #print(self.style, "Plan Prompt...")
        plan = await self.call_llm_async(plan_prompt)

        # ③ SQL 생성
        style_prompt = {
            "default": "",
            "join-first": "Prefer explicit joins before selecting columns.",
            "subquery": "Try solving this with a subquery.",
            "aggregation": "Try solving this by focusing on aggregation logic."
        }.get(self.style, "")

        sql_prompt = f"""You are a SQL expert. Based on the schema, plan, and previous reflections, generate the final, **SQLite-compatible** SQL query. DO NOT use user functions and REGEXP, GLOB, CAST.
Your final answer must be only the SQL query enclosed in "```sql" and "```". Do not include any other text or comments.

# Schema:
{schema_info}

# Plan:
{plan}

# Previous Reflections:
{reflection}

# Style:
{style_prompt}

# Question:
{question}

# SQL Query:
"""
        print(self.style, "Generate SQL...")
        sql = await self.call_llm_async(sql_prompt)
        return plan.strip(), extract_sql_from_response(sql.strip())

    async def call_llm_async(self, prompt, temperature=0.3):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class VerifierAgent:
    def __init__(self, db_path="example.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)

    def verify(self, sql):
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return True, rows
        except Exception as e:
            return False, str(e)

class ReflectionAgent:
    def __init__(self, api_key, base_url, model="solar-pro2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # Synchronous client for simplicity
        self.model_name = model
        self.knowledge_dir = "./knowledge"
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def generate_insights(self, question, schema, golden_sql, predicted_sqls):
        """Use LLM to generate insights considering question, schema, golden_sql, and a list of predicted_sqls."""
        diffs = []
        for i, predicted_sql in enumerate(predicted_sqls):
            diff = difflib.unified_diff(
                golden_sql.splitlines(), predicted_sql.splitlines(),
                fromfile="golden_sql", tofile=f"predicted_sql_{i}", lineterm=""
            )
            diffs.append(f"Diff for Prediction {i}: \n" + "\n".join(diff))
        diff_text = "\n\n".join(diffs)

        prompt = f"""You are a SQL reflection expert. Given the question, DB schema, golden SQL, and multiple predicted SQLs (from different agent styles), analyze why the predictions might be incorrect or suboptimal collectively. 
Consider the question's intent (e.g., aggregation, joins) and schema constraints (e.g., table relationships, column types), and identify common patterns or differences across predictions.
Provide 3-5 concise, general bullet-point lessons that apply broadly to similar text-to-SQL tasks, avoiding any references to specific tables, columns, question phrases, or schema details. Focus on abstract patterns like join strategies, aggregation handling, or syntax pitfalls across multiple approaches.
Do not output anything else.

# Question: {question}

# Schema: {schema}

# Golden SQL: {golden_sql}

# Predicted SQLs: {', '.join(predicted_sqls)}

# SQL Diffs: {diff_text}

# Bullet-point lessons:
"""
        response = self.call_llm(prompt)
        # Parse response for bullets
        insights = [line.strip() for line in response.splitlines() if line.strip().startswith("- ")]
        return insights

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5  # Moderate temperature for insightful but consistent outputs
        )
        return res.choices[0].message.content.strip()

    def merge_reflections(self, new_insights):
        """Merge new insights with the latest reflection file."""
        latest_reflection = ""
        knowledge_dir = self.knowledge_dir
        files = [f for f in os.listdir(knowledge_dir) if f.startswith("reflection_") and f.endswith(".txt")]
        if files:
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(knowledge_dir, x)))
            with open(os.path.join(knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
                latest_reflection = f.read()

        # Parse existing insights
        existing_insights = []
        if latest_reflection:
            existing_insights = [line.strip() for line in latest_reflection.splitlines() if line.strip().startswith("- ")]

        # Merge and deduplicate insights
        merged_insights = list(dict.fromkeys(existing_insights + new_insights))
        return merged_insights

    def save_reflection(self, question, schema, golden_sql, predicted_sqls):
        """Save reflection insights to a timestamped file, handling a list of predicted_sqls."""
        if not isinstance(predicted_sqls, list):
            predicted_sqls = [predicted_sqls]  # Fallback for single prediction
        insights = self.generate_insights(question, schema, golden_sql, predicted_sqls)
        merged_insights = self.merge_reflections(insights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(self.knowledge_dir, f"reflection_{timestamp}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(merged_insights))
        return merged_insights