import inspect

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import (
    SEED,
    BERT_MODEL_NAME,
    MAX_LEN,
    NUM_EPOCHS,
    BERT_OUTPUT_DIR,
)
from .utils import (
    set_seed,
    split_nyt,
    evaluate,
    append_result,
)


class NYTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def make_training_args():
    kwargs = dict(
        output_dir=str(BERT_OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    # 兼容不同 transformers 版本：eval_strategy / evaluation_strategy
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    # 兼容不同 transformers 版本：logging_dir 可能已移除
    if "logging_dir" in params:
        kwargs["logging_dir"] = str(BERT_OUTPUT_DIR / "logs")

    # 只保留当前版本支持的参数，避免 TypeError
    kwargs = {k: v for k, v in kwargs.items() if k in params}

    return TrainingArguments(**kwargs)


def main():
    set_seed(SEED)

    train, val, test, le = split_nyt()
    num_labels = len(le.classes_)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_labels,
    )

    train_ds = NYTDataset(train["text"], train["label_id"], tokenizer)
    val_ds = NYTDataset(val["text"], val["label_id"], tokenizer)
    test_ds = NYTDataset(test["text"], test["label_id"], tokenizer)

    args = make_training_args()

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 兼容不同 transformers 版本：processing_class / tokenizer
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)

    metrics = evaluate(test["label_id"], preds)
    print(f"Task3-BERT: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")

    append_result("Task3-BERT", metrics["accuracy"], metrics["macro_f1"])


if __name__ == "__main__":
    main()