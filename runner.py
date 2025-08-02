import asyncio
from agents import SQLPlanGeneratorAgent, VerifierAgent, ReflectionAgent
from utils import load_dataset
import os
from dotenv import load_dotenv
import argparse

async def main(idx, example, schema):
    print(f"Running example {idx}...")

    question, db_id, gold_sql = example["question"], example["db_id"], example["query"]
    
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY_0')
    base_url = os.getenv('UPSTAGE_API_BASE')
    model = "solar-pro2"
    
    # 1. Multiple SQL Generator Agents with different strategies
    print("🧠 Generating SQL...")      
    agent_styles = ["default", "join-first", "subquery", "aggregation"]
    
    # Create agents and use them as async context managers
    agents = [
        SQLPlanGeneratorAgent(style=style, api_key=api_key, base_url=base_url, model=model) 
        for style in agent_styles
    ]

    print("🧠 Generating plans and SQL in parallel...")
    
    # 2. 병렬적으로 SQL 생성 with proper context management
    async def run_agent(agent, question, schema):
        async with agent:
            return await agent.generate_full_sql_async(question, schema)
    
    sql_outputs = await asyncio.gather(
        *[run_agent(agent, question, schema) for agent in agents]
    )

    # 3. 후보 구성
    candidates = list(zip([f"Agent-{agent.style}" for agent in agents], sql_outputs))
    
    # 4. Verification        
    db_path=f"./datasets/spider/database/{db_id}/{db_id}.sqlite"    
    verifier = VerifierAgent(db_path=db_path)
    
    valid_gold, gold_sql_value = verifier.verify(gold_sql)
    print(f"🧠 Gold SQL: {gold_sql_value[:40]}")
    print(f"🧠 Gold SQL: {gold_sql}")
    
    # Collect all predicted SQLs
    predicted_sqls = [sql for _, sql in sql_outputs]    
    
    # 5. Process candidates (verification and logging)
    print("🧠 Reflections...")    
    reflection_agent = ReflectionAgent(api_key=api_key, base_url=base_url, model=model)
    results = []
    for name, (_, sql) in candidates:
        ok, sql_value = verifier.verify(sql)
        if ok and sql_value == gold_sql_value:
            answer = "✅ Correct"
        else:
            answer = "❌ Incorrect"

        print(f"✅ Executable: {ok}, {name}, SQL: {sql}")               
        print(f" {answer}  : Result Sample: {str(sql_value)[:100]}") 
        if ok:
            results.append((name, sql, sql_value, ok))
    
    # 6. Single reflection per problem, after collecting all predictions
    accumulated_insights = reflection_agent.save_reflection(question, schema, gold_sql, predicted_sqls)
    
    # print(f"🧠 Gold SQL: {gold_sql}")
    # print(f"🧠 Predicted SQLs: {predicted_sqls}")
            
    # Save reflection for each candidate, now including question and schema         
    print(f"🧠 Accumulated Insights:")
    for insight in accumulated_insights:
        print(f"  {insight}")
    print("--------------------------------\n")
                                                        
    return sql_value == gold_sql_value, "ret_error"
                                                

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Spider dataset evaluation with configurable start index and number of examples.")
    parser.add_argument("--start", type=int, default=95, help="Starting index for dataset (default: 95)")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to process (default: 5)")
    args = parser.parse_args()

    # Validate arguments
    if args.start < 0:
        raise ValueError("Start index (--start) must be non-negative.")
    if args.n <= 0:
        raise ValueError("Number of examples (--n) must be positive.")
        
    # 데이터셋 로드
    dataset, schemas = load_dataset('./datasets/spider', 'dev')    

    # Validate dataset bounds
    if args.start >= len(dataset):
        raise ValueError(f"Start index ({args.start}) exceeds dataset size ({len(dataset)}).")
    if args.start + args.n > len(dataset):
        print(f"Warning: Requested {args.n} examples from index {args.start}, but dataset has only {len(dataset) - args.start} remaining. Adjusting n.")
        args.n = len(dataset) - args.start

    start_idx = args.start
    n_examples = args.n
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