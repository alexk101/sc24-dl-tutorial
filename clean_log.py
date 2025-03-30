from pathlib import Path

def clean_log(log_path: Path, output: Path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    clean_lines = []
    for line in lines:
        if 'W323' not in line:
            clean_lines.append(line)
    with open(output, 'w') as f:
        f.writelines(clean_lines)


if __name__ == '__main__':
    log_path = Path('param-sweep-python-3229738.out')
    output = Path('param-sweep-clean.out')
    clean_log(log_path, output)
