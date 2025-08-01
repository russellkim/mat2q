START=0
CNT=200
WORKERS=1

echo "===== Generating SQLs ====="
#python src/main.py --method rule-based --num-workers $WORKERS --indices-file results/testsuite_error_indices.txt
python src/main.py --method llm --num-workers $WORKERS --indices-file results/testsuite_error_indices.txt


echo "===== Preparing eval subset ====="
LATEST=$(ls -t results/*.tsv 2>/dev/null | head -1)
echo "Using $LATEST"
python prepare_eval_subset.py --indices results/testsuite_error_indices.txt --pred-tsv $LATEST

echo "===== Evaluating ====="
python test-suite-sql-eval/evaluation.py --gold results/gold_subset.txt --pred results/pred_subset.txt --db datasets/spider/database/ --table datasets/spider/tables.json --etype exec --plug_value

echo "Error indices:"
head results/testsuite_error_indices.txt
