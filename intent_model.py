import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pickle
import os
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model/intent_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_intent_data(data_path):
    texts = []
    labels = []
    label2idx = {}
    idx2label = {}
    current_idx = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            texts.append(text)
            if label not in label2idx:
                label2idx[label] = current_idx
                idx2label[current_idx] = label
                current_idx += 1
            labels.append(label2idx[label])

    return texts, labels, label2idx, idx2label

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    return accuracy, avg_loss

def train(model, train_loader, val_loader, optimizer, device, num_epochs, model_save_path):
    model.train()
    best_val_acc = 0.0
    accumulation_steps = 4  # 梯度累积步数
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        # 训练阶段
        model.train()
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()*accumulation_steps:.4f}')
        
        train_acc = 100 * correct / total if total > 0 else 0
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        
        # 验证阶段
        val_acc, val_loss = evaluate(model, val_loader, device)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            logging.info(f'Model saved with validation accuracy: {val_acc:.2f}%')

def augment_text(text):
    # 替换词
    replacements = {
        "这部电影": ["这部影片", "这部片子", "这部作品", "这部片子", "这部影片"],
        "是谁": ["是哪个", "是哪些", "是哪些人", "是哪些人", "是哪些人"],
        "什么": ["哪些", "什么类型", "什么种类", "什么类别", "什么类型"],
        "多少": ["多大", "多高", "多长", "多短", "多宽"],
        "哪里": ["什么地方", "哪个地方", "哪些地方", "哪个位置", "哪些位置"],
        "什么时候": ["何时", "几时", "什么时候", "什么时间", "什么时间"]
    }
    
    # 随机替换
    for old, news in replacements.items():
        if old in text:
            text = text.replace(old, np.random.choice(news))
    
    return text

if __name__ == '__main__':
    # 参数设置
    bert_model_name = 'model/chinese-roberta-wwm-ext'
    max_len = 128
    batch_size = 4
    num_epochs = 30  # 增加训练轮数
    learning_rate = 5e-6  # 降低学习率
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data_path = 'data/intent_data.txt'
    texts, labels, label2idx, idx2label = load_intent_data(data_path)

    # 数据增强
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        # 原始样本
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # 简单替换
        augmented_texts.append(text.replace("这部电影", "这部影片"))
        augmented_labels.append(label)
        augmented_texts.append(text.replace("这部电影", "这部片子"))
        augmented_labels.append(label)
        
        # 随机替换
        for _ in range(3):  # 每个样本生成3个变体
            augmented_text = augment_text(text)
            if augmented_text != text:  # 确保生成了不同的变体
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

    texts = augmented_texts
    labels = augmented_labels

    # 保存标签映射
    with open('data/label2idx.pkl', 'wb') as f:
        pickle.dump(label2idx, f)
    with open('data/idx2label.pkl', 'wb') as f:
        pickle.dump(idx2label, f)

    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = IntentClassifier(bert_model_name, len(label2idx)).to(device)

    # 创建数据集
    dataset = IntentDataset(texts, labels, tokenizer, max_len)
    
    # 使用随机种子确保可重复性
    torch.manual_seed(42)
    
    # 计算每个类别的样本数量
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logging.info(f"类别分布: {label_counts}")
    
    # 创建训练集和验证集的索引
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    # 训练模型
    model_save_path = 'model/intent_classifier.pth'
    train(model, train_loader, val_loader, optimizer, device, num_epochs, model_save_path) 