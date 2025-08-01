import asyncio
from agents import SQLPlanGeneratorAgent, VerifierAgent
from utils import load_dataset
import os
from dotenv import load_dotenv

async def main(idx, example, schema):
    print(f"Running example {idx}...")

    question, db_id, gold_sql = example["question"], example["db_id"], example["query"]
    
    # print("🧠 Schema:", schema)
    
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY_0')
    base_url = os.getenv('UPSTAGE_API_BASE')
    model = "solar-pro2"
    
    
    # 1. Multiple SQL Generator Agents with different strategies
    # 1-1. 여러 스타일의 SQL 에이전트 생성
    print("🧠 Generating SQL...")      
    agent_styles = ["default", "join-first", "subquery", "aggregation"]
    #agent_styles = ["default"]    
    
    # Create agents and use them as async context managers
    agents = [
        SQLPlanGeneratorAgent(style=style, api_key=api_key, base_url=base_url, model=model) 
        for style in agent_styles
    ]

    print("🧠 Generating plans and SQL in parallel...")
    
    # 2 병렬적으로 SQL 생성 with proper context management
    async def run_agent(agent, question, schema):
        async with agent:
            return await agent.generate_full_sql_async(question, schema)
    
    sql_outputs = await asyncio.gather(
        *[run_agent(agent, question, schema) for agent in agents]
    )

    # 3. 후보 구성
    candidates = list(zip([f"Agent-{agent.style}" for agent in agents], sql_outputs))
    #print(f"🧠 Candidates: {candidates}")
    
    # 4. Verification        
    db_path=f"./datasets/spider/database/{db_id}/{db_id}.sqlite"    
    verifier = VerifierAgent(db_path=db_path)
    
    valid_gold, gold_sql_value = verifier.verify(gold_sql)
    print(f"🧠 Gold SQL: {gold_sql_value[:40]}")
    print(f"🧠 Gold SQL: {gold_sql}")
        
    results = []
    for name, (_, sql) in candidates:
        ok, sql_value = verifier.verify(sql)
        #print(f"\n{name} SQL:\n{sql}")
        if ok and sql_value == gold_sql_value:
            answer= "✅ Correct"
        else:
            answer= "❌ Incorrect"

        print(f"✅ Executable: {ok}, {name}, SQL: {sql}")               
        print(f" {answer}  : Result Sample: {str(sql_value)[:100]}") 
        if ok:
            results.append((name, sql, sql_value, ok))            
                                                        
    return sql_value == gold_sql_value, "ret_error"

if __name__ == "__main__":
    # 데이터셋 로드
    dataset, schemas = load_dataset('./datasets/spider', 'dev')    

    start_idx = 95
    n_examples = 5   
    total_correct = 0
    total_error = {}
    total_error['miss'] = 0
    for idx in range(start_idx, start_idx + n_examples):
        example = dataset[idx]
        schema = schemas[example["db_id"]]
        bCorrect, error = asyncio.run(main(idx, example, schema))
        if bCorrect:
            total_correct += 1
            
    print(f"Total correct: {total_correct}/{n_examples}")
