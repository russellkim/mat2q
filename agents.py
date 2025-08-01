import openai
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from utils import extract_sql_from_response

class PlannerAgent:
    def __init__(self, api_key, base_url, model="gpt-3.5-turbo"):
        self.api_key=api_key
        self.base_url=base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model

    def plan(self, question, schema_info):
        prompt = f"""You are a planner. Given the question and schema, decompose into SQL plan steps. Do not use user functions and REGEXP, GLOB, CAST. 

# Question: 
{question}

#Schema Info: 
{schema_info}

# Give high-level steps to generate the SQL with smart and concise."""
        return self.call_llm(prompt)

    def call_llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()

class SQLGeneratorAgent:
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

    def get_prompt(self, plan, question, schema_info):
        style_prompt = {
            "default": "Efficient and Simple SQL query",
            "join-first": "Prefer explicit joins before selecting columns.",
            "subquery": "If possible, try solving this with a subquery.",
            "aggregation": "Try solving this by focusing on aggregation logic."
        }.get(self.style, "")        
        
        prompt = f"""You are a SQL expert. Based on the schema and the plan, generate the final, **SQLite-compatible** SQL query. Do not use user functions and REGEXP, GLOB, CAST. 
Your final answer must be only the SQL query enclosed in "```sql" and "```". Do not include any other text or comments.

# Schema: 
{schema_info}

# Plan: 
{plan}

# Style:
{style_prompt}

# Question: 
{question}

# SQL Query:
"""
        print(self.style, prompt)
        print("--------------------------------")
        return prompt

    async def generate_sql_async(self, plan, question, schema_info):
        prompt = self.get_prompt(plan, question, schema_info)
        return await self.call_llm_async(prompt)

    async def call_llm_async(self, prompt):
        res = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return extract_sql_from_response(res.choices[0].message.content.strip())


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

    async def generate_full_sql_async(self, question, schema_info):
        # ① plan 생성
        plan_prompt = f""""You are a planner. Given the question and schema, decompose into SQL plan steps. DO NOT use user functions and REGEXP, GLOB, CAST.

# Question:
{question}

#Schema Info:
{schema_info}

# Give high-level steps to generate the SQL with smart and concise."""
        print(self.style, "Plan Prompt...")
        plan = await self.call_llm_async(plan_prompt)

        # ② SQL 생성
        style_prompt = {
            "default": "",
            "join-first": "Prefer explicit joins before selecting columns.",
            "subquery": "Try solving this with a subquery.",
            "aggregation": "Try solving this by focusing on aggregation logic."
        }.get(self.style, "")

        sql_prompt = f"""You are a SQL expert. Based on the schema and the plan, generate the final, **SQLite-compatible** SQL query. DO NOT use user functions and REGEXP, GLOB, CAST.
Your final answer must be only the SQL query enclosed in "```sql" and "```". Do not include any other text or comments.

# Schema:
{schema_info}

# Plan:
{plan}

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
