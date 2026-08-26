# 环境配置与依赖导入
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from pathlib import Path
# 按需自行添加
import math
import logging

# 日志配置：同时输出到控制台和日志文件
LOG_DIR = Path(__file__).resolve().parent / "outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "run.log"

logger = logging.getLogger("nlp_experiment")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 文件日志
file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 超参数
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-4
N_HEAD = 4
# embed_dim must be divisible by num_heads
NUM_LAYERS = 6

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "IMDB_datasets" / "hf_imdb"
TOKENIZER_DIR = ROOT / "tokenizer-bert-base-uncased"
OUTPUT_DIR = ROOT / "outputs"


# 数据加载与预处理
def prepare_data():
    # 加载数据集
    train_data = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    test_data = pd.read_parquet(DATA_DIR / "test-00000-of-00001.parquet")

    # 将DataFrame格式数据中text列提取为文字列表
    train_list = train_data["text"].to_list()
    test_list = test_data["text"].to_list()

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    # 标记化文本并将其转换为输入格式
    train_encodings = tokenizer(train_list, padding=True, max_length=512,
                                truncation=True, return_tensors='pt')
    test_encodings = tokenizer(test_list, padding=True, max_length=512,
                                truncation=True, return_tensors='pt')

    # 标签
    train_labels_list = train_data["label"].tolist()
    train_labels = torch.tensor(train_labels_list)
    test_labels_list = test_data["label"].tolist()
    test_labels = torch.tensor(test_labels_list)

    # 构建 DataLoader
    train_dataset = TensorDataset(train_encodings['input_ids'],
                                  train_encodings['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'],
                                   test_encodings['attention_mask'], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 分类数（num_classes）：标签的唯一值数量
    num_classes = len(train_data["label"].unique())

    # 词汇表大小（vocab_size）：BERT分词器的词汇表大小
    vocab_size = tokenizer.vocab_size

    return train_loader, test_loader, num_classes, vocab_size


# 简单位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)].permute(1, 0, 2)


# Transformer模型定义
class TransformerSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, output_dim=100):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.pos_encoder = PositionalEncoding(output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.classifier = nn.Linear(output_dim, 2)  # 用于训练的辅助分类头

    def forward(self, input_ids, attention_mask):
        # 嵌入层
        x = self.embedding(input_ids) * math.sqrt(self.output_dim)
        x = x + self.pos_encoder(x)

        # Transformer编码
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        attn_mask = (attention_mask == 0)  # 转换为Transformer需要的mask格式
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)

        # 句子向量提取（取第一个token）
        sentence_vector = x[:, 0, :]
        projected = self.projection(sentence_vector)
        sentence_vector = torch.tanh(self.layer_norm(projected))

        # 分类输出
        logits = self.classifier(sentence_vector)
        return sentence_vector, logits


# 模型参数
nhead = N_HEAD
num_layers = NUM_LAYERS

# 数据准备
train_loader, test_loader, num_classes, vocab_size = prepare_data()
logger.info(f"数据加载完成，分类数={num_classes}，词汇表大小={vocab_size}")

# 验证数据加载器是否正确
sample_batch = next(batch for batch in train_loader)
input_ids, attention_mask, labels = sample_batch
logger.info(f"训练数据批次形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, labels={labels.shape}")


def easy_test_model():
    # 在训练前初步验证模型实现代码
    model = TransformerSentenceEncoder(output_dim=100, vocab_size=vocab_size).to(device)
    logger.info("开始测试模型前向传播...")
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        sentence_vector, logits = model(input_ids, attention_mask)
        logger.info(f"句子向量形状: {sentence_vector.shape}, logits形状: {logits.shape}")
        logger.info(f"句子向量前5个值: {sentence_vector[0, :5].tolist()}")
        logger.info(f"logits: {logits[0].tolist()}")
        break


easy_test_model()


# 模型训练
def train_model(vector_dim=100):
    model = TransformerSentenceEncoder(output_dim=vector_dim, vocab_size=vocab_size).to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 预训练阶段
    logger.info(f"开始训练 vector_dim={vector_dim}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            _, logits = model(input_ids, attention_mask)

            # 损失函数
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        torch.cuda.empty_cache()
        logger.info(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
    return model


# 模型评估
def eval_model(vector_dim=100):
    train_features = []
    test_features = []
    train_labels_list = []
    test_labels_list = []
    model = train_model(vector_dim)
    logger.info(f"开始评估 vector_dim={vector_dim}...")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            vectors, _ = model(input_ids, attention_mask)
            train_features.append(vectors.cpu())
            train_labels_list.append(labels.cpu())
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            vectors, _ = model(input_ids, attention_mask)
            test_features.append(vectors.cpu())
            test_labels_list.append(labels.cpu())
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels_list, dim=0).numpy()
    test_features = torch.cat(test_features, dim=0).numpy()
    test_labels = torch.cat(test_labels_list, dim=0).numpy()
    # 归一化特征
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    # 逻辑回归分类器
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(train_features, train_labels)
    test_preds = lr_clf.predict(test_features)
    # 评估结果
    test_accuracy = accuracy_score(test_labels, test_preds)
    return test_accuracy


# 执行实验
if __name__ == "__main__":
    start_time = time.time()
    acc_100 = eval_model(100)
    logger.info(f"100-dim Model Accuracy: {acc_100:.4f}")
    acc_200 = eval_model(200)
    logger.info(f"200-dim Model Accuracy: {acc_200:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / "score.txt"
    with open(score_path, "a+", encoding='utf-8') as f:
        f.write("\nResult Comparison:\n")
        f.write(f"100-dim Model Accuracy: {acc_100:.4f}\n")
        f.write(f"200-dim Model Accuracy: {acc_200:.4f}\n")
        f.write(f"cost time: {total_time:.4f} seconds\n")

    logger.info(f"\nResult Comparison:\n100-dim Model Accuracy: {acc_100:.4f}\n200-dim Model Accuracy: {acc_200:.4f}\ncost time: {total_time:.4f} seconds")
