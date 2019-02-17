__author__ = 'yingyuankai'


def read_next(input_file):
    """
    read per line
    """
    with open(input_file, "r", encoding='utf8') as f:
        for line in f:
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            yield line
