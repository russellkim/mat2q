import asyncio
from agents import SQLPlanGeneratorAgent, VerifierAgent, ReflectionAgent, SelectorAgent, SelectorReflectionAgent, KnowledgeSummarizerAgent
from utils import load_dataset
import os
from dotenv import load_dotenv
import argparse

async def main(idx, example, schema, api_key, base_url, model):
    print(f"Running example {idx}...")

    question, db_id, gold_sql = example["question"], example["db_id"], example["query"]
        
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
                                                        
    return style_correct_dict, selector_correct, "ret_error"

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Run Spider dataset evaluation with configurable start index, number of examples, and summarization interval.")
    parser.add_argument("--start", type=int, default=95, help="Starting index for dataset (default: 95)")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to process (default: 5)")
    parser.add_argument("--summary-interval", type=int, default=10, help="Interval for summarizing reflections (default: 10)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY_0')
    base_url = os.getenv('UPSTAGE_API_BASE')
    model = "solar-pro2"

    if args.start < 0:
        raise ValueError("Start index (--start) must be non-negative.")
    if args.n <= 0:
        raise ValueError("Number of examples (--n) must be positive.")
    if args.summary_interval <= 0:
        raise ValueError("Summary interval (--summary-interval) must be positive.")

    dataset, schemas = load_dataset('./datasets/spider', 'dev')    

    if args.start >= len(dataset):
        raise ValueError(f"Start index ({args.start}) exceeds dataset size ({len(dataset)}).")
    if args.start + args.n > len(dataset):
        print(f"Warning: Requested {args.n} examples from index {args.start}, but dataset has only {len(dataset) - args.start} remaining. Adjusting n.")
        args.n = len(dataset) - args.start

    start_idx = args.start
    n_examples = args.n
    summary_interval = args.summary_interval
    
    agent_styles = ["default", "join-first", "subquery", "aggregation"]
    style_correct_counts = {style: 0 for style in agent_styles}
    selector_correct_count = 0
    total_correct = 0
    total_error = {'miss': 0}
    summarization_count = 0  # Track summarizations

    summarizer_agent = KnowledgeSummarizerAgent(api_key=api_key, base_url=base_url, model=model)
    for idx in range(start_idx, start_idx + n_examples):
        example = dataset[idx]
        schema = schemas[example["db_id"]]
        style_correct_dict, selector_correct, error = asyncio.run(main(idx, example, schema, api_key, base_url, model))
        
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
        
        # Summarize if at interval
        if (idx - start_idx + 1) % summary_interval == 0:
            print(f"🧠 Summarizing reflections at example {idx}...")
            base_summary = summarizer_agent.summarize_reflections(is_selector=False)
            selector_summary = summarizer_agent.summarize_reflections(is_selector=True)
            summarization_count += 1
            print(f"🧠 Base Agent Summary:")
            for section, insights in base_summary.items():
                print(f"  {section.capitalize()}:")
                for insight in insights:
                    print(f"    {insight}")
            print(f"🧠 Selector Summary:")
            for insight in selector_summary["general"]:
                print(f"  {insight}")
            print("--------------------------------\n")
    
    print(f"Total examples where at least one style correct: {total_correct}/{n_examples}")
    print("\nPer-Style Performance Statistics:")
    for style in agent_styles:
        correct = style_correct_counts[style]
        accuracy = (correct / n_examples) * 100 if n_examples > 0 else 0
        print(f"Style '{style}': {correct}/{n_examples} ({accuracy:.2f}%)")
    
    print("\nSelector Performance Statistics:")
    selector_accuracy = (selector_correct_count / n_examples) * 100 if n_examples > 0 else 0
    print(f"Selector: {selector_correct_count}/{n_examples} ({selector_accuracy:.2f}%)")
    
    print(f"\nSummarization Events: {summarization_count}")