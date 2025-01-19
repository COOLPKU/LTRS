当然可以，以下是一个可以中英文切换的README.md文件示例：

```markdown
# LTRS: Improving Word Sense Disambiguation via Learning to Rank Senses

[中文](#中文) | [English](#english)

## 中文

### 作者:
- 王涵思
- 王悦
- 梁启良
- 刘洋

### 单位:
- 北京大学多媒体信息处理国家重点实验室
- 北京大学计算机科学学院

### 联系方式:
- wanghansi2019@pku.edu.cn
- wyy209@pku.edu.cn
- lql.pkucs@gmail.com
- liuyang@pku.edu.cn

---

### 摘要
词义消歧 (WSD) 是精确语义理解的基础任务。传统的训练策略通常仅考虑目标词的预定义词义，并从相对有限的实例中学习每个词义，忽略了相似词义的影响。为了解决这些问题，我们提出了学习排序词义 (LTRS) 的方法来增强该任务。该方法帮助模型通过对扩展的词义定义列表进行排序，从更广泛的实例中学习表示和消歧词义。通过使用LTRS，我们的模型在中文WSD中达到了79.6%的SOTA F1分数，并在低资源环境中表现出稳健性。此外，它表现出优异的训练效率，比以前的方法收敛速度更快。这为WSD提供了一种新的技术方法，也可以应用于其他语言的任务。

### 简介
词义消歧 (WSD) 旨在识别上下文中的单词词义，这对于精确的语义理解至关重要，并有助于多种下游应用，如信息检索、文本摘要和机器翻译。近年来，将词汇知识（如词义定义）集成到神经架构中，成功地提高了监督WSD方法的性能。

### 方法
LTRS的总体思路是帮助模型根据目标词的语义相似性对定义进行排序。我们的方法的整体架构包括上下文编码器和定义编码器，两者都使用预训练模型BERT初始化。模型被鼓励根据目标词的语义相似性对词义定义进行排序，并通过包括其他单词的定义来扩展候选定义列表。

### 结果
通过使用LTRS，我们的模型在中文WSD中超过了以前的顶级模型，并在低资源环境中表现出稳健性。此外，它还比以前的方法表现出更好的训练效率。

---

### 如何使用代码

#### 要求
- Python
- PyTorch
- Transformers
- SentenceTransformers
- tqdm

#### 安装
1. 克隆仓库：
```bash
git clone https://github.com/COOLPKU/LTRS.git
cd LTRS
```

2. 安装所需的包：
```bash
pip install -r requirements.txt
```

#### 训练模型
使用以下命令训练模型：
```bash
python main.py --mode train --train_data_path ./data/MF_train_data.tsv --dev_data_path ./data/MF_dev_data.tsv --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --bge_model_path ./bge_model --ckpt ./output --device cuda:0
```

#### 评估模型
使用以下命令评估模型：
```bash
python main.py --mode evaluate --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --ckpt ./output --device cuda:0
```

#### 参数
- `--rand_seed`: 随机种子以确保结果可重复。
- `--grad_norm`: 梯度裁剪的最大范数。
- `--lr`: 学习率。
- `--warmup`: 学习率预热步数。
- `--context_max_len`: 上下文的最大长度。
- `--gloss_max_len`: 释义的最大长度。
- `--epochs`: 训练的轮数。
- `--gloss_bsz`: 释义的批处理大小。
- `--gradient_accumulation_step`: 梯度累积的步数。
- `--encoder-name`: 编码器的名称。
- `--train_data_path`: 训练数据的路径。
- `--test_data_path`: 测试数据的路径。
- `--dev_data_path`: 开发数据的路径。
- `--word_sense_path`: 词义数据的路径。
- `--bge_model_path`: BGE模型的路径。
- `--ckpt`: 模型检查点的保存路径。
- `--device`: 训练设备，例如cuda:0或cpu。
- `--tau1`, `--tau2`, `--tau3`: 超参数tau1、tau2和tau3的值。
- `--steps`: 训练的总步数。
- `--loss_fn_type`: 损失函数的类型，例如listnet或list_mle。
- `--mode`: 运行模式，例如train或evaluate。

#### 示例用法
```bash
python main.py --mode train --train_data_path ./data/MF_train_data.tsv --dev_data_path ./data/MF_dev_data.tsv --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --bge_model_path ./bge_model --ckpt ./output --device cuda:0
```

#### 引用
如果你使用了这个代码，请引用我们的论文：
```bibtex
@inproceedings{wang2025ltrs,
  title={LTRS: Improving Word Sense Disambiguation via Learning to Rank Senses},
  author={Wang, Hansi and Wang, Yue and Liang, Qiliang and Liu, Yang},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

#### 论文
有关更多详细信息，请参阅我们的[论文](https://example.com/your-paper-link)。

#### 致谢
本文得到了国家自然科学基金（编号62036001）和国家社会科学基金（编号18ZDA295）的支持。

---

## English

### Authors:
- Hansi Wang
- Yue Wang
- Qiliang Liang
- Yang Liu

### Affiliation:
- National Key Laboratory for Multimedia Information Processing, Peking University
- School of Computer Science, Peking University

### Contact:
- wanghansi2019@pku.edu.cn
- wyy209@pku.edu.cn
- lql.pkucs@gmail.com
- liuyang@pku.edu.cn

---

### Abstract
Word Sense Disambiguation (WSD) is a fundamental task critical for accurate semantic understanding. Conventional training strategies usually only consider predefined senses for target words and learn each of them from relatively limited instances, neglecting the influence of similar ones. To address these problems, we propose the method of Learning to Rank Senses (LTRS) to enhance the task. This method helps a model learn to represent and disambiguate senses from a broadened range of instances via ranking an expanded list of sense definitions. By employing LTRS, our model achieves a SOTA F1 score of 79.6% in Chinese WSD and exhibits robustness in low-resource settings. Moreover, it shows excellent training efficiency, achieving faster convergence than previous methods. This provides a new technical approach to WSD and may also apply to the task for other languages.

### Introduction
Word Sense Disambiguation (WSD) aims to identify the sense of words in context, which is critical for accurate semantic understanding and beneficial to multiple downstream applications, such as Information Retrieval, Text Summarization, and Machine Translation. In recent years, integrating lexical knowledge, such as sense definitions, within neural architectures has successfully enhanced the performance of supervised WSD methods.

### Methodology
The general idea of LTRS is to help a model learn to rank definitions based on their semantic similarity with the target word. The overall architecture of our method involves a context encoder and a definition encoder, both initialized with the pre-trained model BERT. The model is encouraged to rank sense definitions according to their semantic similarity with the target word, and the candidate definition list is expanded by including definitions from other words.

### Results
By employing LTRS, our model outperforms previous top-performing models in Chinese WSD and exhibits robustness in low-resource settings. Furthermore, it also achieves better training efficiency than the previous methods.

---

### How to Use the Code

#### Requirements
- Python
- PyTorch
- Transformers
- SentenceTransformers
- tqdm

#### Installation
1. Clone the repository:
```bash
git clone https://github.com/COOLPKU/LTRS.git
cd LTRS
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

#### Training the Model
To train the model, use the following command:
```bash
python main.py --mode train --train_data_path ./data/MF_train_data.tsv --dev_data_path ./data/MF_dev_data.tsv --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --bge_model_path ./bge_model --ckpt ./output --device cuda:0
```

#### Evaluating the Model
To evaluate the model, use the following command:
```bash
python main.py --mode evaluate --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --ckpt ./output --device cuda:0
```

#### Arguments
- `--rand_seed`: Random seed for reproducibility.
- `--grad_norm`: Maximum norm for gradient clipping.
- `--lr`: Learning rate.
- `--warmup`: Number of warmup steps for learning rate.
- `--context_max_len`: Maximum length of the context.
- `--gloss_max_len`: Maximum length of the gloss.
- `--epochs`: Number of training epochs.
- `--gloss_bsz`: Batch size for gloss.
- `--gradient_accumulation_step`: Number of steps for gradient accumulation.
- `--encoder-name`: Name of the encoder.
- `--train_data_path`: Path to the training data.
- `--test_data_path`: Path to the test data.
- `--dev_data_path`: Path to the development data.
- `--word_sense_path`: Path to the word sense data.
- `--bge_model_path`: Path to the BGE model.
- `--ckpt`: Path to save the model checkpoint.
- `--device`: Device to run the training on, e.g., cuda:0 or cpu.
- `--tau1`, `--tau2`, `--tau3`: Values of hyperparameters tau1, tau2, and tau3.
- `--steps`: Total number of training steps.
- `--loss_fn_type`: Type of loss function, e.g., listnet or list_mle.
- `--mode`: Mode to run, e.g., train or evaluate.

#### Example Usage
```bash
python main.py --mode train --train_data_path ./data/MF_train_data.tsv --dev_data_path ./data/MF_dev_data.tsv --test_data_path ./data/MF_dev_data.tsv --word_sense_path ./data --bge_model_path ./bge_model --ckpt ./output --device cuda:0
```

#### Citation
If you use this code, please cite our paper:
```bibtex
@inproceedings{wang2025ltrs,
  title={LTRS: Improving Word Sense Disambiguation via Learning to Rank Senses},
  author={Wang, Hansi and Wang, Yue and Liang, Qiliang and Liu, Yang},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

#### Paper
For more details, please refer to our [paper](https://example.com/your-paper-link).

#### Acknowledgements
This paper is supported by the National Natural Science Foundation of China (No. 62036001) and the National Social Science Foundation of China (No. 18ZDA295).

---

For more details, please refer to our [paper](https://example.com/your-paper-link).
```

将`https://example.com/your-paper-link`替换为你的论文的实际链接。这样，读者可以根据需要切换阅读中文或英文内容。
