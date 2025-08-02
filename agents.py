import openai
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from utils import extract_sql_from_response
import difflib
from datetime import datetime
import json
import re

class SQLPlanGeneratorAgent:
    def __init__(self, api_key, base_url, model="gpt-3.5-turbo", style="default"):
        self.api_key = api_key
        self.base_url = base_url
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
        if self.client:
            await self.client.close()

    def get_latest_reflection(self):
        """Read the most recent reflection JSON from the knowledge directory."""
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
        reflection = self.get_latest_reflection()
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
        latest_reflections = {"general": [], "default": [], "join-first": [], "subquery": [], "aggregation": []}
        files = [f for f in os.listdir(self.knowledge_dir) if f.startswith("reflection_") and f.endswith(".json")]
        if files:
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.knowledge_dir, x)))
            with open(os.path.join(self.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
                latest_reflections = json.load(f)
        for key in latest_reflections:
            combined = latest_reflections[key] + new_insights.get(key, [])
            latest_reflections[key] = list(dict.fromkeys(combined))
        return latest_reflections

    def save_reflection(self, question, schema, golden_sql, styled_predictions):
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
        self.knowledge_dir = "./knowledge/selector"
        self.prompt_file = "prompt/prompt_selector.txt"
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.prompt_file), exist_ok=True)
        # Initialize prompt file if not exists
        if not os.path.exists(self.prompt_file):
            initial_prompt = """You are a SQL selector expert. Given the question, schema, candidate SQLs from different styles, and previous selector reflections, choose the best SQL. 
Consider correctness (if executable, prefer matching intent), efficiency (simpler queries preferred), and alignment with question intent. Use reflections to avoid past mistakes.
Output only in this format:
Selected Style: <style>
Selected SQL: <sql>

# Question: {question}

# Schema: {schema}

# Candidates: {candidates_str}

# Previous Selector Reflections: {reflection}
"""
            with open(self.prompt_file, 'w', encoding='utf-8') as f:
                f.write(initial_prompt)

    def get_latest_selector_reflection(self):
        files = [f for f in os.listdir(self.knowledge_dir) if f.startswith("selector_reflection_") and f.endswith(".json")]
        if not files:
            return ""
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.knowledge_dir, x)))
        with open(os.path.join(self.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            reflections = json.load(f)
        general = reflections.get("general", [])
        return "# Previous Selector Reflections:\n" + "\n".join(general)

    def select_sql(self, question, schema, candidates):
        reflection = self.get_latest_selector_reflection()
        candidates_str = "\n".join([f"Style: {style}, SQL: {sql}" for style, sql in candidates])
        # Read prompt from file
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            question=question,
            schema=schema,
            candidates_str=candidates_str,
            reflection=reflection
        )
        response = self.call_llm(prompt)
        lines = response.splitlines()
        selected_style = lines[0].split(": ")[1] if lines else candidates[0][0]
        selected_sql = lines[1].split(": ")[1] if len(lines) > 1 else candidates[0][1]
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
        diff = difflib.unified_diff(
            golden_sql.splitlines(), selected_sql.splitlines(),
            fromfile="golden_sql", tofile="selected_sql", lineterm=""
        )
        diff_text = "\n".join(diff)
        outcome = "correct" if was_correct else "incorrect"
        prompt = f"""You are a selector reflection expert. Given the question, schema, golden SQL, selected SQL, and whether the selection was correct, analyze why the choice led to an {outcome} outcome. 
Provide 3-5 general bullet-point lessons for better future selections, focusing on abstract patterns (e.g., when to prefer join-based styles). Avoid specific table/column names.
Output only bullet-point lessons starting with "- ".

# Question: {question}

# Schema: {schema}

# Golden SQL: {golden_sql}

# Selected Style: {selected_style}

# Selected SQL: {selected_sql}

# Outcome: {outcome}

# SQL Diff: {diff_text}
"""
        response = self.call_llm(prompt)
        insights = [line.strip() for line in response.splitlines() if line.strip().startswith("- ")]
        return {"general": insights}

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
            latest_reflections[key] = list(dict.fromkeys(combined))
        return latest_reflections

    def save_reflection(self, question, schema, golden_sql, selected_style, selected_sql, was_correct):
        insights = self.generate_insights(question, schema, golden_sql, selected_style, selected_sql, was_correct)
        merged_insights = self.merge_reflections(insights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = os.path.join(self.knowledge_dir, f"selector_reflection_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(merged_insights, f, indent=4)
        return merged_insights

class SelectorAgentTeacher:
    def __init__(self, api_key, base_url, model="solar-pro2"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model
        self.knowledge_dir = "./knowledge/selector"
        self.prompt_file = "prompt/prompt_selector.txt"
        self.prompt_backup_prefix = "prompt_selector_"
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.prompt_file), exist_ok=True)

    def get_latest_selector_reflection(self):
        files = [f for f in os.listdir(self.knowledge_dir) if f.startswith("selector_reflection_") and f.endswith(".json")]
        if not files:
            return {"general": []}, None
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.knowledge_dir, x)))
        with open(os.path.join(self.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            reflections = json.load(f)
        return reflections, latest_file

    def rewrite_prompt(self):
        """Backup current prompt, rewrite based on reflections, deduplicate, and save new reflection."""
        reflections, latest_file = self.get_latest_selector_reflection()
        general_reflections = reflections.get("general", [])
        reflections_str = "\n".join(general_reflections)

        # Backup current prompt
        if os.path.exists(self.prompt_file):
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                current_prompt = f.read()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            backup_filename = os.path.join(os.path.dirname(self.prompt_file), f"{self.prompt_backup_prefix}{timestamp}.txt")
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(current_prompt)
            print(f"🧠 Backed up current selector prompt to {backup_filename}")

        # Read current prompt (for rewrite)
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            current_prompt = f.read()

        prompt = f"""You are a prompt optimization expert. Given the current selector prompt and accumulated selector reflections, rewrite the prompt to incorporate key lessons for better SQL selection. 
Ensure the rewritten prompt remains concise, improves selection accuracy, and retains the output format: 'Selected Style: <style>\nSelected SQL: <sql>'.
Avoid specific table/column names; focus on general patterns (e.g., prioritizing joins for multi-table queries).

# Current Prompt:
{current_prompt}

# Reflections:
{reflections_str}

# Output only the rewritten prompt:
"""
        response = self.call_llm(prompt)

        # Save rewritten prompt
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(response)

        # Deduplicate reflections and save new file
        deduped_reflections = {"general": list(dict.fromkeys(general_reflections))}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        new_filename = os.path.join(self.knowledge_dir, f"selector_reflection_{timestamp}.json")
        with open(new_filename, 'w', encoding='utf-8') as f:
            json.dump(deduped_reflections, f, indent=4)

        return response, deduped_reflections

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return res.choices[0].message.content.strip()