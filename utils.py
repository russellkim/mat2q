from datasets import load_dataset
import os
import json
import sqlite3
from typing import List, Dict
import re

def load_spider_example(idx=0):
    dataset = load_dataset("spider", split="validation")
    example = dataset[idx]
    return example["question"], example["db_id"]

def load_spider_data(self, spider_path: str) -> List[Dict]:
    """Load Spider dataset"""
    data_files = [
        "train_spider.json",
        "dev.json",
        "train_others.json"
    ]

    all_data = []
    for file_name in data_files:
        file_path = os.path.join(spider_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)

    return all_data

def load_schema(db_id):
    # 간단화를 위해 dummy schema 사용
    # 실제는 spider/tables.json에서 불러오기
    return "Table: singer(id, name), concert(id, singer_id, venue, date)"

def load_sqlite_db(db_id):
    # 실제 Spider DB를 복사하거나 sqlite로 마운트 필요
    return f"./databases/spider{db_id}/{db_id}.sqlite"

# def extract_sql_from_response(response_text):
#     matches = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL)
#     if matches:
#         return matches[-1]
#     # Fallback: Find any SELECT-like string
#     sql_match = re.search(r'SELECT.*?(?:;|FROM|WHERE|GROUP|ORDER|LIMIT|$)', response_text, re.DOTALL | re.IGNORECASE)
#     return sql_match.group(0) if sql_match else None

def extract_sql_from_response(response: str) -> str:
    """
    Extracts the SQL query from the LLM's response.
    Handles cases where the response might include markdown code blocks.
    """
    # Look for a SQL code block
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, strip whitespace and hope for the best
    return response.strip()

def load_dataset(data_dir, split='dev'):
    """
    Spider 데이터셋과 각 DB의 정확한 멀티-테이블 스키마를 로드합니다.
    - 각 테이블의 CREATE TABLE 구문을 정확히 생성합니다.
    - 각 테이블의 첫 번째 데이터 행을 샘플로 함께 제공하여 LLM의 이해를 돕습니다.
    """
    print(f"Loading Spider dataset ({split} split)...")

    dataset_path = os.path.join(data_dir, f'{split}.json')
    tables_path = os.path.join(data_dir, 'tables.json')
    database_path = os.path.join(data_dir, 'database')

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    schemas = {}
    for db_info in tables_data: # 각 db_info는 DB 하나 전체에 대한 정보를 담고 있음
        db_id = db_info['db_id']
        db_file_path = os.path.join(database_path, db_id, f'{db_id}.sqlite')

        # Step 1: 테이블별로 정보를 재구성
        tables = []
        for table_name in db_info['table_names_original']:
            tables.append({
                'name': table_name,
                'columns': [],
                'pk': []
            })

        # 테이블 인덱스를 기준으로 컬럼들을 각 테이블에 분배
        for i, (table_idx, col_name) in enumerate(db_info['column_names_original']):
            if table_idx < 0:  # '*'와 같은 특수 컬럼은 제외
                continue
            col_type = db_info['column_types'][i]
            tables[table_idx]['columns'].append((col_name, col_type))

        # Primary Key 정보를 각 테이블에 분배
        for pk_col_index in db_info.get('primary_keys', []):
            table_idx, col_name = db_info['column_names_original'][pk_col_index]
            tables[table_idx]['pk'].append(col_name)

        # Group foreign keys by FK table
        fk_by_table = {idx: [] for idx in range(len(tables))}
        foreign_keys = db_info.get('foreign_keys', [])
        for fk in foreign_keys:
            pk_table_idx, pk_col = db_info['column_names_original'][fk[0]]
            fk_table_idx, fk_col = db_info['column_names_original'][fk[1]]
            ref_table = tables[pk_table_idx]['name']
            fk_by_table[fk_table_idx].append((fk_col, ref_table, pk_col))

        # Step 2: 재구성된 정보를 바탕으로 스키마 문자열 생성
        schema_parts = []
        for table_data in tables:
            table_name = table_data['name']

            # 컬럼 정의 생성
            col_defs = [f"`{col_name}` {col_type.upper()}" for col_name, col_type in table_data['columns']]

            # Primary Key 정의 추가
            if table_data['pk']:
                pk_defs = [f"`{pk_col}`" for pk_col in table_data['pk']]
                col_defs.append(f"PRIMARY KEY ({', '.join(pk_defs)})")

            # Foreign Keys for this table
            for fk_col, ref_table, ref_col in fk_by_table[table_idx]:
                col_defs.append(f"FOREIGN KEY (`{fk_col}`) REFERENCES `{ref_table}`(`{ref_col}`)")

            create_statement = f"CREATE TABLE `{table_name}` (\n  " + ",\n  ".join(col_defs) + "\n);"
            schema_parts.append(create_statement)

            # DB에 연결하여 첫 번째 행 데이터 가져오기
            if os.path.exists(db_file_path):
                try:
                    conn = sqlite3.connect(db_file_path)
                    cursor = conn.cursor()
                    cursor.execute(f'SELECT * FROM `{table_name}` LIMIT 1;')
                    row = cursor.fetchone()
                    if row:
                        headers = [desc[0] for desc in cursor.description]
                        row_values = ' | '.join(map(str, row))
                        row_comment = f"-- 1 row from `{table_name}` table:\n-- " + ' | '.join(headers) + f"\n-- {row_values}"
                        schema_parts.append(row_comment)
                    conn.close()
                except Exception as e:
                    print(f"Warning: Could not fetch row from {db_id}.{table_name}. Error: {e}")

        schemas[db_id] = "\n\n".join(schema_parts)

    print("Dataset and schemas loaded successfully.")
    return dataset, schemas
