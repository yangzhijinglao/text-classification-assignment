import argparse

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from .config import SEED, W2V_AG_PATH, W2V_NYT_PATH
from .utils import (
    set_seed,
    split_nyt,
    tokenize,
    evaluate,
    append_result,
    load_glove,
    doc_vec_from_embedding,
    doc_vec_from_w2v,
    load_ag_texts,
)


def train_w2v(sentences, save_path):
    """训练 100 维 Word2Vec。"""
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=5,
        seed=SEED,
    )
    model.save(str(save_path))
    return model


def get_or_train_w2v_ag():
    """AG News 训练 Word2Vec，若已存在则直接加载。"""
    if W2V_AG_PATH.exists():
        return Word2Vec.load(str(W2V_AG_PATH))

    texts = load_ag_texts()
    sentences = [tokenize(t) for t in texts]
    return train_w2v(sentences, W2V_AG_PATH)


def get_or_train_w2v_nyt(train_texts):
    """NYT 训练集训练 Word2Vec，若已存在则直接加载。"""
    if W2V_NYT_PATH.exists():
        return Word2Vec.load(str(W2V_NYT_PATH))

    sentences = [tokenize(t) for t in train_texts]
    return train_w2v(sentences, W2V_NYT_PATH)


def train_clf(X_train, y_train, X_test, y_test, name):
    """使用 Logistic Regression 训练并评价。"""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    metrics = evaluate(y_test, pred)
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    append_result(name, metrics["accuracy"], metrics["macro_f1"])


def run_glove():
    """Task2.1 Pre-trained GloVe 100d。"""
    set_seed(SEED)

    train, val, test, le = split_nyt()
    embeddings = load_glove()

    X_train = np.vstack([
        doc_vec_from_embedding(tokenize(t), embeddings)
        for t in train["text"]
    ])
    X_test = np.vstack([
        doc_vec_from_embedding(tokenize(t), embeddings)
        for t in test["text"]
    ])

    train_clf(
        X_train,
        train["label_id"],
        X_test,
        test["label_id"],
        "Task2-GloVe-100d",
    )


def run_w2v_ag():
    """Task2.2 Word2Vec trained on AG News。"""
    set_seed(SEED)

    train, val, test, le = split_nyt()
    model = get_or_train_w2v_ag()

    X_train = np.vstack([
        doc_vec_from_w2v(tokenize(t), model)
        for t in train["text"]
    ])
    X_test = np.vstack([
        doc_vec_from_w2v(tokenize(t), model)
        for t in test["text"]
    ])

    train_clf(
        X_train,
        train["label_id"],
        X_test,
        test["label_id"],
        "Task2-Word2Vec-AG",
    )


def run_w2v_nyt():
    """Task2.3 Word2Vec trained on NYT。"""
    set_seed(SEED)

    train, val, test, le = split_nyt()
    model = get_or_train_w2v_nyt(train["text"].tolist())

    X_train = np.vstack([
        doc_vec_from_w2v(tokenize(t), model)
        for t in train["text"]
    ])
    X_test = np.vstack([
        doc_vec_from_w2v(tokenize(t), model)
        for t in test["text"]
    ])

    train_clf(
        X_train,
        train["label_id"],
        X_test,
        test["label_id"],
        "Task2-Word2Vec-NYT",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["glove", "ag", "nyt", "all"],
        default="all",
        help="选择要运行的 Word2Vec 实验",
    )
    args = parser.parse_args()

    if args.method in ["glove", "all"]:
        run_glove()

    if args.method in ["ag", "all"]:
        run_w2v_ag()

    if args.method in ["nyt", "all"]:
        run_w2v_nyt()