# 作业一实验报告

## 一、实验目的

本次实验旨在理解和掌握不同文本表示方法在文本分类任务中的基本原理与实际效果。具体目标包括：

1. 理解不同文本表示方法的基本思想；
2. 掌握文本分类任务的基本实验流程；
3. 熟悉 Logistic Regression 在文本分类中的使用；
4. 掌握 Word2Vec、GloVe 等词向量的基本使用方法；
5. 熟悉 BERT 类预训练语言模型的 Fine-tuning 流程；
6. 能够使用 Accuracy 和 Macro-F1 对文本分类模型进行评价；
7. 能够对不同文本表示方法的性能进行基本比较和分析。

---

## 二、数据集

### 2.1 NYT 新闻数据集

NYT 数据集包含两列：

- `text`：新闻文本
- `label`：新闻所属类别

NYT 数据集作为本次实验的主要文本分类数据集，最终模型性能统一在 NYT Test Set 上进行评价。

### 2.2 AG News 数据集

AG News 数据集提供新闻文本，在本次实验中主要用于训练 Word2Vec 词向量。

### 2.3 数据划分

首先对 NYT 数据集进行随机打乱，统一设置随机种子为 `42`，划分比例如下：

| 数据集 | 比例 |
|---|---:|
| Training Set | 80% |
| Validation Set | 10% |
| Test Set | 10% |

所有实验均使用相同的数据划分，以保证不同方法之间具有可比性。

标签编码仅在训练集上 `fit`，验证集和测试集使用训练集的编码器进行转换，避免数据泄漏。

---

## 三、评价指标

### 3.1 Accuracy

即分类正确的样本数量占全部测试样本数量的比例：

$$
\text{Accuracy} = \frac{\text{正确分类样本数}}{\text{总测试样本数}}
$$

### 3.2 Macro-F1

分别计算每一个类别的 F1 Score，然后对所有类别的 F1 进行平均：

$$
\text{Macro-F1} = \frac{1}{C} \sum_{i=1}^{C} F1_i
$$

其中 $C$ 为类别数。对于类别分布可能不均衡的文本分类任务，Macro-F1 可以更加全面地反映模型在不同类别上的分类能力。

---

## 四、实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python 版本 | 3.13.2 |
| 虚拟环境 | `renwu1` |
| 主要依赖 | pandas、numpy、scikit-learn、nltk、gensim、torch、transformers、accelerate |
| 分词工具 | NLTK `word_tokenize` |
| 词向量维度 | 100 |
| 随机种子 | 42 |

运行方式统一为在项目根目录下使用模块方式运行：

```bash
python -m src.task1_bow
python -m src.task2_word2vec --method all
python -m src.task3_bert
```

---

## 五、Task 1：Bag of Words

本部分使用两种 Bag-of-Words 文本表示方法训练文本分类器，统一使用 Logistic Regression 作为分类模型。

### 5.1 Binary Bag of Words

#### 方法原理

首先构建词表 Vocabulary。假设整个训练语料的词表大小为 $|V|$，则每一篇文档被表示为一个 $|V|$ 维向量。对于词表中的第 $i$ 个单词：

$$
\mathbf{x}_i =
\begin{cases}
1, & \text{若该词出现在文档中} \\
0, & \text{否则}
\end{cases}
$$

本方法只考虑一个单词是否出现，而不考虑它出现了多少次。

#### 实现细节

- 分词：使用 `nltk.word_tokenize`，并统一转为小写。
- 向量化：使用 `sklearn.feature_extraction.text.CountVectorizer`，设置 `binary=True`。
- 分类器：`LogisticRegression(max_iter=1000)`。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9887 |
| Macro-F1 | 0.9738 |

### 5.2 Word Frequency

#### 方法原理

仍然使用词表中的所有单词构建文档向量，但与 Binary Bag of Words 不同，本方法记录每个单词在当前文档中的出现次数：

$$
\mathbf{x} = (c_1, c_2, \dots, c_{|V|})
$$

其中 $c_i$ 表示词表中第 $i$ 个词在当前文档中的出现次数。

#### 实现细节

- 分词方式与 5.1 相同。
- 向量化：使用 `CountVectorizer`，设置 `binary=False`。
- 分类器：`LogisticRegression(max_iter=1000)`。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9879 |
| Macro-F1 | 0.9737 |

### 5.3 Task 1 小结

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| Binary BOW | 0.9887 | 0.9738 |
| Word Frequency | 0.9879 | 0.9737 |

两种 BOW 方法表现非常接近，Binary BOW 略高于 Word Frequency。这说明在 NYT 新闻分类任务中，“词是否出现”往往比“词出现多少次”更具有区分度。

---

## 六、Task 2：Word2Vec

本部分使用基于 Word Embedding 的方式表示文档，统一使用 100 维词向量，分类器仍然使用 Logistic Regression。

对于一篇包含 $n$ 个有效单词的文档，假设对应词向量为 $\mathbf{e}(w_1), \mathbf{e}(w_2), \dots, \mathbf{e}(w_n)$，则文档向量通过所有词向量的平均得到：

$$
\mathbf{e}(d) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{e}(w_i)
$$

### 6.1 Pre-trained GloVe

#### 方法原理

使用公开的预训练 GloVe Embedding（glove.6B），选择 100 维 GloVe word vectors 作为单词表示。

#### 实现细节

- 加载 `glove.6B.100d.txt`。
- 对每篇 NYT 文档分词，找到每个单词对应的 GloVe 向量。
- 对文档内所有有效单词的 Embedding 取平均，得到 100 维文档向量。
- 使用 Logistic Regression 完成分类。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 待填 |
| Macro-F1 | 待填 |

### 6.2 Word2Vec Trained on AG News

#### 方法原理

使用 AG News 的文本数据自行训练一个 100 维 Word2Vec Model。

#### 实现细节

- 使用 `gensim.models.Word2Vec` 训练。
- 参数设置：`vector_size=100`，`window=5`，`min_count=1`，`epochs=5`，`seed=42`。
- 对每篇 NYT 文档分词，找到每个单词对应的 Word2Vec Embedding。
- 对文档内所有有效单词的 Embedding 取平均，得到 100 维文档向量。
- 使用 Logistic Regression 完成分类。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 待填 |
| Macro-F1 | 待填 |

### 6.3 Word2Vec Trained on NYT

#### 方法原理

使用 NYT 训练集文本数据训练 100 维 Word2Vec Model，随后使用同样的方法计算文档表示，再使用 Logistic Regression 训练分类器。

#### 实现细节

- 仅在 NYT 训练集上训练 Word2Vec，避免使用验证集和测试集信息。
- 其余流程与 6.2 相同。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 待填 |
| Macro-F1 | 待填 |

### 6.4 Task 2 小结

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| GloVe-100d | 待填 | 待填 |
| Word2Vec-AG | 待填 | 待填 |
| Word2Vec-NYT | 待填 | 待填 |

词向量平均法通常会弱于 BOW 和 BERT，因为平均操作会丢失词序信息和关键词权重信息。具体表现以实际运行结果为准。

---

## 七、Task 3：Pre-trained Neural Models

本部分使用预训练语言模型直接完成文本分类任务，微调 BERT，并在 NYT Test Set 上报告 Accuracy 和 Macro-F1。

### 7.1 BERT

#### 方法原理

使用 `google-bert/bert-base-uncased` 进行文本分类 Fine-tuning。

#### 训练要求

| 配置项 | 值 |
|---|---|
| Pre-trained model | `google-bert/bert-base-uncased` |
| Maximum sequence length | 64 |
| Number of epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Weight decay | 0.01 |
| 最佳模型选择指标 | Macro-F1 |

#### 实现细节

- 使用 `BertTokenizerFast` 对文本进行编码，最大长度 64。
- 使用 `BertForSequenceClassification`，分类头根据 NYT 类别数初始化。
- 使用 HuggingFace `Trainer` 进行训练。
- 每个 epoch 后在验证集上评估，根据 Macro-F1 保存最优模型。
- 训练完成后在测试集上进行预测。

#### 训练过程

| Epoch | Eval Loss | Eval Accuracy | Eval Macro-F1 |
|---|---:|---:|---:|
| 1 | 0.1007 | 0.9731 | 0.9373 |
| 2 | 0.1055 | 0.9757 | 0.9415 |
| 3 | 0.1264 | 0.9731 | 0.9341 |

训练总时长约 4203 秒（CPU 环境）。

#### 实验结果

| 指标 | 值 |
|---|---:|
| Accuracy | 0.9792 |
| Macro-F1 | 0.9537 |

---

## 八、结果汇总

| 方法 | 文本表示 | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Binary BOW | 词袋（0/1） | 0.9887 | 0.9738 |
| Word Frequency | 词袋（词频） | 0.9879 | 0.9737 |
| GloVe-100d | 预训练词向量平均 | 待填 | 待填 |
| Word2Vec-AG | AG 训练词向量平均 | 待填 | 待填 |
| Word2Vec-NYT | NYT 训练词向量平均 | 待填 | 待填 |
| BERT-base-uncased | 预训练语言模型微调 | 0.9792 | 0.9537 |

---

## 九、结果分析与讨论

### 9.1 BOW 表现分析

从已有结果可以看出，Binary BOW 和 Word Frequency 在 NYT 数据集上都取得了很高的性能（Accuracy 均超过 0.98）。这说明：

- NYT 新闻分类任务中，类别与关键词高度相关；
- 即使不考虑词序和上下文，仅靠词袋表示就能达到很好的分类效果；
- Binary BOW 略高于 Word Frequency，说明“词是否出现”比“词出现多少次”更具区分度。

### 9.2 BERT 表现分析

BERT 的 Accuracy 为 0.9792，Macro-F1 为 0.9537，略低于 BOW。可能原因包括：

1. 数据集本身较容易，BOW 已经接近性能上限；
2. 最大序列长度仅为 64，较长新闻被截断，丢失部分信息；
3. 仅训练 3 个 epoch，未做超参数搜索；
4. 在 CPU 上训练，未使用 GPU 加速，可能影响训练充分性；
5. BERT 的分类头是随机初始化的，需要更多数据或更长时间收敛。

### 9.3 词向量方法预期分析

词向量平均法通常会弱于 BOW 和 BERT，原因在于：

- 平均操作会丢失词序信息；
- 所有词被同等对待，无法突出关键词；
- 未登录词（OOV）会被忽略，造成信息损失。

### 9.4 方法对比总结

| 方法类别 | 优点 | 缺点 |
|---|---|---|
| BOW | 简单、快速、可解释性强 | 丢失词序，维度高，稀疏 |
| Word2Vec 平均 | 稠密、低维、语义信息 | 丢失词序，平均削弱关键词 |
| BERT | 上下文感知，语义表达强 | 计算成本高，需要微调 |

---

## 十、代码运行说明

### 10.1 环境安装

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

### 10.2 数据准备

将 `nyt.csv` 和 `ag.csv` 放入 `data/` 目录。

下载 GloVe：

```text
http://nlp.stanford.edu/data/glove.6B.zip
```

解压后，将 `glove.6B.100d.txt` 放入：

```text
data/glove.6B.100d.txt
```

### 10.3 运行实验

在项目根目录下执行：

```bash
python -m src.task1_bow
python -m src.task2_word2vec --method all
python -m src.task3_bert
```

### 10.4 查看结果

实验结果保存在：

```text
outputs/results.csv
```

查看命令：

```bash
python -c "import pandas as pd; print(pd.read_csv('outputs/results.csv').to_string(index=False))"
```

---

## 十一、结论

本次实验分别使用 Bag-of-Words、Word2Vec / GloVe 词向量以及 BERT 预训练语言模型完成 NYT 新闻文本分类任务，并使用 Accuracy 和 Macro-F1 进行评价。

实验结果表明：

1. BOW + Logistic Regression 在该数据集上表现非常强，Binary BOW 的 Accuracy 达到 0.9887；
2. BERT 微调后 Accuracy 为 0.9792，Macro-F1 为 0.9537；
3. 不同文本表示方法各有优缺点，实际应用中需要根据任务特点、数据规模和计算资源进行选择。

通过本次实验，加深了对文本表示方法和文本分类流程的理解，掌握了从传统词袋模型到预训练语言模型的完整实验方法。

---

## 十二、附录：项目结构

```text
text-classification-assignment/
├── data/
│   ├── nyt.csv
│   ├── ag.csv
│   └── glove.6B.100d.txt
├── outputs/
│   └── results.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── task1_bow.py
│   ├── task2_word2vec.py
│   ├── task3_bert.py
│   └── run_all.py
├── requirements.txt
├── .gitignore
└── README.md
```

GitHub 仓库链接：

```text
https://github.com/yangzhijinglao/text-classification-assignment
```