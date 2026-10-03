from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .config import SEED
from .utils import (
    set_seed,
    split_nyt,
    tokenize,
    evaluate,
    append_result,
)


def run_bow(binary: bool, name: str):
    set_seed(SEED)

    train, val, test, le = split_nyt()

    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        binary=binary,
        lowercase=False,  # tokenize 中已经转小写
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train["label_id"])

    pred = clf.predict(X_test)

    metrics = evaluate(test["label_id"], pred)
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    append_result(name, metrics["accuracy"], metrics["macro_f1"])


if __name__ == "__main__":
    # Task 1.1 Binary Bag of Words
    run_bow(binary=True, name="Task1-Binary BOW")

    # Task 1.2 Word Frequency
    run_bow(binary=False, name="Task1-Word Frequency")