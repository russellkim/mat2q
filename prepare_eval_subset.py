import json
import argparse
import csv
import os

def main():
    parser = argparse.ArgumentParser(description='Prepare subset gold and pred files for test-suite-sql-eval.')
    parser.add_argument('--dev-json', default='datasets/spider/dev.json', help='Path to Spider dev.json.')
    parser.add_argument('--pred-tsv', default='results/predicted_sql.txt', help='Path to predicted_sql.txt TSV.')
    parser.add_argument('--start', type=int, default=0, help='Starting index (0-based).')
    parser.add_argument('--n', type=int, default=None, help='Number of items (default: all from start).')
    parser.add_argument('--indices', type=str, default=None, help='Comma-separated indices or file path.')
    parser.add_argument('--gold-out', default='results/gold_subset.txt', help='Output gold file.')
    parser.add_argument('--pred-out', default='results/pred_subset.txt', help='Output pred file.')
    args = parser.parse_args()

    # Load dev data
    with open(args.dev_json, 'r') as f:
        dev_data = json.load(f)

    # Load predictions as dict by Index
    preds = {}
    if os.path.exists(args.pred_tsv):
        with open(args.pred_tsv, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                idx = int(row['Index'])
                preds[idx] = row['Predicted_SQL']
    else:
        print(f"Warning: {args.pred_tsv} not found - pred file will be empty.")

    # Get indices
    if args.indices:
        if os.path.isfile(args.indices):
            with open(args.indices, 'r') as f:
                indices_str = f.read().strip()
        else:
            indices_str = args.indices
        try:
            indices = sorted([int(i.strip()) for i in indices_str.split(',')])
        except ValueError:
            print("Invalid indices format. Exiting.")
            return
    else:
        start = max(0, args.start)
        end = len(dev_data) if args.n is None else min(start + args.n, len(dev_data))
        indices = list(range(start, end))

    # Generate files
    with open(args.gold_out, 'w') as gold_f, open(args.pred_out, 'w') as pred_f:
        for idx in indices:
            if idx >= len(dev_data):
                continue
            item = dev_data[idx]
            gold_f.write(f"{item['query']}\t{item['db_id']}\n")
            pred_sql = preds.get(idx, '')  # Empty if no prediction
            pred_f.write(f"{idx}\t{pred_sql}\n")  # Include original index

    print(f"Generated {len(indices)} items:")
    print(f"- {args.gold_out}")
    print(f"- {args.pred_out}")
    print("\nRun evaluation:")
    print(f"python test-suite-sql-eval/evaluation.py --gold {args.gold_out} --pred {args.pred_out} --db datasets/spider/database/ --table datasets/spider/tables.json --etype exec")
    print("(Add --plug_value if needed)")

if __name__ == '__main__':
    main() 
