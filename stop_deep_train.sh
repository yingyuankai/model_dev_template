ps xf | grep "python -u main_deep.py --config=$1 --mode=train" | grep -v "grep" | awk '{print $1}' | xargs kill -9
ps xf | grep "python -u main_deep.py --config=$1 --mode=eval" | grep -v "grep" | awk '{print $1}' | xargs kill -9

