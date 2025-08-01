## for setting up experiment env. for Text-to-SQL

mkdir plan

## config
cp /Users/jabkim/git/qd-sql/.gitignore .
cp /Users/jabkim/git/qd-sql/.env .

## datasets
mkdir datasets
pushd datasets
ls
ln -s /Users/jabkim/git/spider/spider_data spider
popd

## evaluation
cp -r /Users/jabkim/git/qd-sql/prepare_eval_subset.py .
cp -r /Users/jabkim/git/qd-sql/test-suite-sql-eval .
cp /Users/jabkim/git/CPB-SQL/eval.sh .
cp /Users/jabkim/git/CPB-SQL/eval_idx.sh .
