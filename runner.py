import asyncio
from agents import SQLPlanGeneratorAgent, VerifierAgent, ReflectionAgent, SelectorAgent, SelectorReflectionAgent, SelectorAgentTeacher
from utils import load_dataset
import os
from dotenv import load_dotenv
import argparse
import json

async def main(idx, example, schema, reflection_age_counter, selector_correct_count, total_examples_processed):
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
    db_path = f"./datasets/spider/database/{db_id}/{db_id}.sqlite"    
    verifier = VerifierAgent(db_path=db_path)
    
    valid_gold, gold_sql_value = verifier.verify(gold_sql)
    print(f"🧠 Gold SQL: {gold_sql_value[:40]}")
    print(f"🧠 Gold SQL: {gold_sql}")
    
    # Collect styled predictions as dict {style: sql}
    styled_predictions = {agent_styles[i]: sql for i, (_, sql) in enumerate(sql_outputs)}
    
    # 5. Process candidates (verification and logging)
    reflection_agent = ReflectionAgent(api_key=api_key, base_url=base_url, model=model)
    results = []
    style_correct_dict = {}  # Per-example correctness per style
    for name, (_, sql) in candidates:
        ok, sql_value = verifier.verify(sql)
        is_correct = ok and sql_value == gold_sql_value
        if is_correct:
            answer = "✅ Correct"
        else:
            answer = "❌ Incorrect"

        style = name.split("-")[1]
        if style not in agent_styles:
            print(f"Warning: Invalid style '{style}' found for {name}. Skipping.")
            continue
        style_correct_dict[style] = is_correct

        print(f"✅ Executable: {ok}, {name}, SQL: {sql}")               
        print(f" {answer}  : Result Sample: {str(sql_value)[:100]}") 
        if ok:
            results.append((name, sql, sql_value, ok))
    
    # 6. Base reflection
    base_insights = reflection_agent.save_reflection(question, schema, gold_sql, styled_predictions)
    print(f"🧠 Base Agent Accumulated Insights:")
    for section, insights in base_insights.items():
        print(f"  {section.capitalize()}:")
        for insight in insights:
            print(f"    {insight}")
    print("--------------------------------\n")
    
    # 7. Selector Agent: Choose one from candidates
    selector_agent = SelectorAgent(api_key=api_key, base_url=base_url, model=model)
    candidates_list = [(style, sql) for style, sql in styled_predictions.items()]
    selected_style, selected_sql = selector_agent.select_sql(question, schema, candidates_list)
    
    # 8. Verify selected SQL
    ok_selected, selected_value = verifier.verify(selected_sql)
    selector_correct = ok_selected and selected_value == gold_sql_value
    print(f"🧠 Selected Style: {selected_style}, Correct: {selector_correct}")
    print(f"🧠 Selected SQL: {selected_sql}")
    
    # 9. Selector Reflection
    selector_reflection_agent = SelectorReflectionAgent(api_key=api_key, base_url=base_url, model=model)
    selector_insights = selector_reflection_agent.save_reflection(question, schema, gold_sql, selected_style, selected_sql, selector_correct)
    print(f"🧠 Selector Accumulated Insights:")
    for insight in selector_insights["general"]:
        print(f"  {insight}")
    print("--------------------------------\n")
    
    # 10. Check for prompt rewrite
    rewrite_triggered = False
    selector_teacher = SelectorAgentTeacher(api_key=api_key, base_url=base_url, model=model)
    files = [f for f in os.listdir(selector_teacher.knowledge_dir) if f.startswith("selector_reflection_") and f.endswith(".json")]
    if files:
        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(selector_teacher.knowledge_dir, x)))
        with open(os.path.join(selector_teacher.knowledge_dir, latest_file), 'r', encoding='utf-8') as f:
            latest_reflections = json.load(f)
        reflection_count = len(latest_reflections.get("general", []))
        # Compute current selector accuracy
        selector_accuracy = (selector_correct_count / total_examples_processed * 100) if total_examples_processed > 0 else 0
        # Check both size and age
        if reflection_count >= args.rewrite_selector_size and reflection_age_counter >= args.min_reflection_age:
            print(f"🧠 Rewriting selector prompt at example {idx}...")
            new_prompt, deduped_reflections = selector_teacher.rewrite_prompt(selector_accuracy)
            rewrite_triggered = True
            print(f"🧠 New Selector Prompt:\n{new_prompt}")
            print(f"🧠 Reset Selector Reflections to Empty")
            print("--------------------------------\n")
            reflection_age_counter = 0  # Reset counter after rewrite
    
    return style_correct_dict, selector_correct, rewrite_triggered, reflection_age_counter + 1, "ret_error"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spider dataset evaluation with configurable start index, number of examples, selector prompt rewrite size, and minimum reflection age.")
    parser.add_argument("--start", type=int, default=95, help="Starting index for dataset (default: 95)")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to process (default: 5)")
    parser.add_argument("--rewrite-selector-size", type=int, default=10, help="Threshold for number of selector reflection items to trigger prompt rewrite (default: 10)")
    parser.add_argument("--min-reflection-age", type=int, default=5, help="Minimum number of examples since last reflection reset to trigger prompt rewrite (default: 5)")
    args = parser.parse_args()

    if args.start < 0:
        raise ValueError("Start index (--start) must be non-negative.")
    if args.n <= 0:
        raise ValueError("Number of examples (--n) must be positive.")
    if args.rewrite_selector_size <= 0:
        raise ValueError("Rewrite selector size (--rewrite-selector-size) must be positive.")
    if args.min_reflection_age <= 0:
        raise ValueError("Minimum reflection age (--min-reflection-age) must be positive.")

    dataset, schemas = load_dataset('./datasets/spider', 'dev')    

    if args.start >= len(dataset):
        raise ValueError(f"Start index ({args.start}) exceeds dataset size ({len(dataset)}).")
    if args.start + args.n > len(dataset):
        print(f"Warning: Requested {args.n} examples from index {args.start}, but dataset has only {len(dataset) - args.start} remaining. Adjusting n.")
        args.n = len(dataset) - args.start

    start_idx = args.start
    n_examples = args.n
    
    agent_styles = ["default", "join-first", "subquery", "aggregation"]
    style_correct_counts = {style: 0 for style in agent_styles}
    selector_correct_count = 0
    total_correct = 0
    total_error = {'miss': 0}
    rewrite_count = 0
    reflection_file_count = 0
    reflection_age_counter = 0  # Track examples since last reset
    total_examples_processed = 0

    for idx in range(start_idx, start_idx + n_examples):
        example = dataset[idx]
        schema = schemas[example["db_id"]]
        style_correct_dict, selector_correct, rewrite_triggered, reflection_age_counter, error = asyncio.run(
            main(idx, example, schema, reflection_age_counter, selector_correct_count, total_examples_processed + 1)
        )
        
        # Update stats
        any_correct = False
        for style, is_correct in style_correct_dict.items():
            if is_correct:
                style_correct_counts[style] += 1
                any_correct = True
        if any_correct:
            total_correct += 1
        if selector_correct:
            selector_correct_count += 1
        if rewrite_triggered:
            rewrite_count += 1
        reflection_file_count += 1  # Increment for each example (new reflection file created)
        total_examples_processed += 1
    
    print(f"Total examples where at least one style correct: {total_correct}/{n_examples}")
    print("\nPer-Style Performance Statistics:")
    for style in agent_styles:
        correct = style_correct_counts[style]
        accuracy = (correct / n_examples) * 100 if n_examples > 0 else 0
        print(f"Style '{style}': {correct}/{n_examples} ({accuracy:.2f}%)")
    
    print("\nSelector Performance Statistics:")
    selector_accuracy = (selector_correct_count / n_examples) * 100 if n_examples > 0 else 0
    print(f"Selector: {selector_correct_count}/{n_examples} ({selector_accuracy:.2f}%)")
    
    print(f"\nSelector Prompt Rewrite Events: {rewrite_count}")
    print(f"Total Reflection Files Created: {reflection_file_count}")