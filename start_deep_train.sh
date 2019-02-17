nohup python -u main_deep.py --config=$1 --mode=train --gpus=$2 > train_err_$1.log 2>&1 &
sleep 10
nohup python -u main_deep.py --config=$1 --mode=eval --gpus=$2 > eval_err_$1.log 2>&1 &
