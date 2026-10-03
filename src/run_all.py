import subprocess
import sys

commands = [
    [sys.executable, "-m", "src.task1_bow"],
    [sys.executable, "-m", "src.task2_word2vec", "--method", "all"],
    [sys.executable, "-m", "src.task3_bert"],
]

for cmd in commands:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)