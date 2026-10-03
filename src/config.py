from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"

# 输出目录
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件
NYT_PATH = DATA_DIR / "nyt.csv"
AG_PATH = DATA_DIR / "ag.csv"
GLOVE_PATH = DATA_DIR / "glove.6B.100d.txt"

# 模型保存路径
W2V_AG_PATH = OUTPUT_DIR / "word2vec_ag.model"
W2V_NYT_PATH = OUTPUT_DIR / "word2vec_nyt.model"
BERT_OUTPUT_DIR = OUTPUT_DIR / "bert"

# 结果文件
RESULTS_PATH = OUTPUT_DIR / "results.csv"

# 随机种子
SEED = 42

# 数据划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# BERT 配置
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64
NUM_EPOCHS = 3