import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .config import (
    NYT_PATH,
    AG_PATH,
    GLOVE_PATH,
    RESULTS_PATH,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)

_NLTK_READY = False


def ensure_nltk():
    """确保 NLTK 分词资源已下载。"""
    global _NLTK_READY
    if _NLTK_READY:
        return

    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource)
            except Exception:
                pass

    _NLTK_READY = True


def set_seed(seed: int = SEED):
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def tokenize(text: str):
    """统一分词：小写 + NLTK word_tokenize。"""
    ensure_nltk()
    return word_tokenize(str(text).lower())


def load_nyt():
    """读取 NYT 数据集。"""
    df = pd.read_csv(NYT_PATH)

    if "text" in df.columns and "label" in df.columns:
        pass
    else:
        # 默认第一列为 text，第二列为 label
        df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})

    df = df[["text", "label"]].dropna().reset_index(drop=True)
    return df


def split_nyt(df=None, seed: int = SEED):
    """
    统一划分 NYT：
    随机打乱 -> 80% train / 10% val / 10% test。
    标签编码只在训练集上 fit，避免数据泄漏。
    """
    if df is None:
        df = load_nyt()

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()

    le = LabelEncoder()
    train["label_id"] = le.fit_transform(train["label"])
    val["label_id"] = le.transform(val["label"])
    test["label_id"] = le.transform(test["label"])

    return train, val, test, le


def evaluate(y_true, y_pred):
    """计算 Accuracy 和 Macro-F1。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def append_result(method: str, accuracy: float, macro_f1: float, path=RESULTS_PATH):
    """追加实验结果到 outputs/results.csv。"""
    row = pd.DataFrame([{
        "method": method,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }])

    if Path(path).exists():
        old = pd.read_csv(path)
        row = pd.concat([old, row], ignore_index=True)

    row.to_csv(path, index=False)


def load_glove(path=GLOVE_PATH, dim: int = 100):
    """加载 GloVe 词向量。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"GloVe 文件不存在: {path}，请下载 glove.6B.zip 并解压出 glove.6B.100d.txt"
        )

    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vec

    return embeddings


def doc_vec_from_embedding(tokens, embeddings, dim: int = 100):
    """用词向量字典对文档内有效词求平均。"""
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def doc_vec_from_w2v(tokens, model, dim: int = 100):
    """用 gensim Word2Vec 模型对文档内有效词求平均。"""
    vecs = []
    for w in tokens:
        if w in model.wv:
            vecs.append(model.wv[w])

    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vecs, axis=0)


def load_ag_texts():
    """读取 AG News 文本，用于训练 Word2Vec。"""
    df = pd.read_csv(AG_PATH)

    if "text" in df.columns:
        texts = df["text"].dropna().astype(str).tolist()
    else:
        cols = [c for c in ["title", "description"] if c in df.columns]
        if cols:
            texts = df[cols].fillna("").agg(" ".join, axis=1).astype(str).tolist()
        else:
            texts = df.iloc[:, 0].dropna().astype(str).tolist()

    return texts