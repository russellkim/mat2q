import openai
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from utils import extract_sql_from_response
import difflib
from datetime import datetime
import json
import re  # For parsing LLM output

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
        """Read the most recent reflection JSON from the knowledge directory, extract general + style-specific."""
        knowledge_dir = "./knowledge"
        if not os.path.exists(knowledge_dir):
            return ""
        files = [f for f in os.listdir(knowledge_dir) if f.startswith("reflection_") and f.endswith(".json")]
        if not files:
            return ""
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(knowledge_dir, x)))
        with open(os.path.join(knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            reflections = json.load(f)
        
        general = reflections.get("general", [])
        specific = reflections.get(self.style, [])
        combined = (
            "# General Reflections:\n" + "\n".join(general) + "\n\n" +
            f"# {self.style.capitalize()} Style Reflections:\n" + "\n".join(specific)
        )
        return combined.strip()

    async def generate_full_sql_async(self, question, schema_info):
        # ① Read latest reflection (general + style-specific)
        reflection = self.get_latest_reflection()

        # ② Plan 생성
        plan_prompt = f"""You are a planner. Given the question, schema, and previous reflections (general and style-specific), decompose into SQL plan steps. DO NOT use user functions and REGEXP, GLOB, CAST.

# Question:
{question}

# Schema Info:
{schema_info}

# Previous Reflections:
{reflection}

# Give high-level steps to generate the SQL with smart and concise."""
        print(self.style, "Plan Prompt...")
        plan = await self.call_llm_async(plan_prompt)

        # ③ SQL 생성
        style_prompt = {
            "default": "",
            "join-first": "Prefer explicit joins before selecting columns.",
            "subquery": "Try solving this with a subquery.",
            "aggregation": "Try solving this by focusing on aggregation logic."
        }.get(self.style, "")

        sql_prompt = f"""You are a SQL expert. Based on the schema, plan, and previous reflections (general and style-specific), generate the final, **SQLite-compatible** SQL query. DO NOT use user functions and REGEXP, GLOB, CAST.
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
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model
        self.knowledge_dir = "./knowledge"
        os.makedirs(self.knowledge_dir, exist_ok=True)
        self.styles = ["default", "join-first", "subquery", "aggregation"]

    def generate_insights(self, question, schema, golden_sql, styled_predictions):
        """Use LLM to generate general + style-specific insights."""
        diffs = []
        predicted_sqls_str = []
        for i, style in enumerate(self.styles):
            predicted_sql = styled_predictions.get(style, "")
            diff = difflib.unified_diff(
                golden_sql.splitlines(), predicted_sql.splitlines(),
                fromfile="golden_sql", tofile=f"predicted_sql_{style}", lineterm=""
            )
            diffs.append(f"Diff for {style} Style: \n" + "\n".join(diff))
            predicted_sqls_str.append(f"{style} Style: {predicted_sql}")
        diff_text = "\n\n".join(diffs)
        predicted_sqls_text = "\n".join(predicted_sqls_str)

        prompt = f"""You are a SQL reflection expert. Given the question, DB schema, golden SQL, and predicted SQLs from different styles, analyze collectively why predictions might be incorrect or suboptimal. 
Consider question intent and schema constraints, identifying common patterns across styles and style-specific issues.
Output in this exact structure:
- General lessons: (3-5 concise, general bullets applying broadly, avoiding specifics)
- Default style lessons: (1-3 bullets tailored to default style)
- Join-first style lessons: (1-3 bullets tailored to join-first style)
- Subquery style lessons: (1-3 bullets tailored to subquery style)
- Aggregation style lessons: (1-3 bullets tailored to aggregation style)
Focus on abstract patterns; avoid references to specific tables/columns/questions/schemas. Do not output anything else.

# Question: {question}

# Schema: {schema}

# Golden SQL: {golden_sql}

# Predicted SQLs: {predicted_sqls_text}

# SQL Diffs: {diff_text}
"""
        response = self.call_llm(prompt)
        # Parse response into dict
        insights = {"general": [], "default": [], "join-first": [], "subquery": [], "aggregation": []}
        current_key = None
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("- General lessons:"):
                current_key = "general"
            elif line.startswith("- Default style lessons:"):
                current_key = "default"
            elif line.startswith("- Join-first style lessons:"):
                current_key = "join-first"
            elif line.startswith("- Subquery style lessons:"):
                current_key = "subquery"
            elif line.startswith("- Aggregation style lessons:"):
                current_key = "aggregation"
            elif line.startswith("- ") and current_key:
                insights[current_key].append(line)
        return insights

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return res.choices[0].message.content.strip()

    def merge_reflections(self, new_insights):
        """Merge new insights dict with the latest reflection JSON."""
        latest_reflections = {"general": [], "default": [], "join-first": [], "subquery": [], "aggregation": []}
        knowledge_dir = self.knowledge_dir
        files = [f for f in os.listdir(knowledge_dir) if f.startswith("reflection_") and f.endswith(".json")]
        if files:
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(knowledge_dir, x)))
            with open(os.path.join(knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
                latest_reflections = json.load(f)

        # Merge per key: append new, deduplicate
        for key in latest_reflections:
            combined = latest_reflections[key] + new_insights.get(key, [])
            latest_reflections[key] = list(dict.fromkeys(combined))
        return latest_reflections

    def save_reflection(self, question, schema, golden_sql, styled_predictions):
        """Save reflection insights to a timestamped JSON file."""
        insights = self.generate_insights(question, schema, golden_sql, styled_predictions)
        merged_insights = self.merge_reflections(insights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(self.knowledge_dir, f"reflection_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(merged_insights, f, indent=4)
        return merged_insights
    
class SelectorAgent:
    def __init__(self, api_key, base_url, model="solar-pro2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model
        self.knowledge_dir = "./knowledge/selector"  # Separate sub-dir for selector reflections
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def get_latest_selector_reflection(self):
        """Read the most recent selector reflection JSON."""
        files = [f for f in os.listdir(self.knowledge_dir) if f.startswith("selector_reflection_") and f.endswith(".json")]
        if not files:
            return ""
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.knowledge_dir, x)))
        with open(os.path.join(self.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            reflections = json.load(f)
        general = reflections.get("general", [])
        return "# Previous Selector Reflections:\n" + "\n".join(general)

    def select_sql(self, question, schema, candidates):
        """Select the best SQL from candidates using LLM."""
        reflection = self.get_latest_selector_reflection()
        candidates_str = "\n".join([f"Style: {style}, SQL: {sql}" for style, sql in candidates])

        prompt = f"""You are a SQL selector expert. Given the question, schema, candidate SQLs from different styles, and previous reflections on selections, choose the best SQL. 
Consider correctness, efficiency, and alignment with intent. Output only the selected style and SQL in format: Selected Style: <style>\nSelected SQL: <sql>

# Question: {question}

# Schema: {schema}

# Candidates: {candidates_str}

# Previous Reflections: {reflection}
"""
        response = self.call_llm(prompt)
        # Parse response
        lines = response.splitlines()
        selected_style = lines[0].split(": ")[1] if lines else ""
        selected_sql = lines[1].split(": ")[1] if len(lines) > 1 else ""
        return selected_style, selected_sql

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()

class SelectorReflectionAgent:
    def __init__(self, api_key, base_url, model="solar-pro2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model
        self.knowledge_dir = "./knowledge/selector"
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def generate_insights(self, question, schema, golden_sql, selected_style, selected_sql, was_correct):
        """Generate reflections on the selection."""
        diff = difflib.unified_diff(
            golden_sql.splitlines(), selected_sql.splitlines(),
            fromfile="golden_sql", tofile="selected_sql", lineterm=""
        )
        diff_text = "\n".join(diff)

        outcome = "correct" if was_correct else "incorrect"

        prompt = f"""You are a selector reflection expert. Analyze why the selection of {selected_style} style led to an {outcome} outcome. 
Provide 3-5 general bullet-point lessons for better future selections, focusing on abstract patterns (e.g., when to prefer subqueries). Avoid specifics.

# Question: {question}

# Schema: {schema}

# Golden SQL: {golden_sql}

# Selected SQL: {selected_sql}

# SQL Diff: {diff_text}

# Bullet-point lessons:
"""
        response = self.call_llm(prompt)
        insights = [line.strip() for line in response.splitlines() if line.strip().startswith("- ")]
        return {"general": insights}  # Simple structure for selector

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return res.choices[0].message.content.strip()

    def merge_reflections(self, new_insights):
        latest_reflections = {"general": []}
        files = [f for f in os.listdir(self.knowledge_dir) if f.startswith("selector_reflection_") and f.endswith(".json")]
        if files:
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.knowledge_dir, x)))
            with open(os.path.join(self.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
                latest_reflections = json.load(f)

        for key in latest_reflections:
            combined = latest_reflections[key] + new_insights.get(key, [])
            latest_reflections[key] = list(dict.fromkeys(combined))  # Dedup
        return latest_reflections

    def save_reflection(self, question, schema, golden_sql, selected_style, selected_sql, was_correct):
        insights = self.generate_insights(question, schema, golden_sql, selected_style, selected_sql, was_correct)
        merged_insights = self.merge_reflections(insights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(self.knowledge_dir, f"selector_reflection_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(merged_insights, f, indent=4)
        return merged_insights    