ps xf | grep "python service.py" | grep -v "grep" | awk '{print $1}' | xargs kill -9

