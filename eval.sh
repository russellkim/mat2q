START=0
CNT=200
WORKERS=4

#python src/main.py --method rule-based --n 20 --num-workers 4
#python src/main.py --method rule-based --start $START --n $CNT --num-workers $WORKERS
python src/main.py --method llm --start $START --n $CNT --num-workers $WORKERS

LATEST=$(ls -t results/*.tsv 2>/dev/null | head -1)
echo "Using $LATEST"
python prepare_eval_subset.py --start $START --n $CNT --pred-tsv $LATEST

python test-suite-sql-eval/evaluation.py --gold results/gold_subset.txt --pred results/pred_subset.txt --db datasets/spider/database/ --table datasets/spider/tables.json --etype exec --plug_value

echo "Error indices:"
head results/testsuite_error_indices.txt
