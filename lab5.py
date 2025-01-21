import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import chardet
from torch.nn.utils.rnn import pad_sequence
import argparse  


# 命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-modal sentiment model")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--data_path', type=str, default=r"C:\大学\大三\当代人工智能\实验五\实验五数据\data", help='Path to the data directory')
    parser.add_argument('--train_file', type=str, default=r"C:\大学\大三\当代人工智能\实验五\实验五数据\train.txt", help='Path to the training file')
    parser.add_argument('--test_file', type=str, default=r"C:\大学\大三\当代人工智能\实验五\实验五数据\test_without_label.txt", help='Path to the test file')
    return parser.parse_args()

args = parse_args()

# 使用命令行参数进行配置
DATA_PATH = args.data_path
TRAIN_FILE = args.train_file
TEST_FILE = args.test_file
BATCH_SIZE = args.batch_size   
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用 chardet 库自动检测文件的编码格式
def read_file_with_auto_encoding(file_path):
    with open(file_path, "rb") as f:  
        raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]  
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:  
        return f.read()

# 加载数据集
def load_data(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        next(f)  
        for line in f:
            guid, label = line.strip().split(",")
            data.append((guid, label))
    return pd.DataFrame(data, columns=["guid", "label"])

# 定义数据集类,转化为适合输入神经网络的形式
class MultiModalDataset(Dataset):
    def __init__(self, data, data_dir, tokenizer, transform):
        self.data = data
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, label = self.data.iloc[idx]
        # 文本处理（读取对应的{guid}.txt文件中的内容并转化为张量）
        text_file = os.path.join(self.data_dir, f"{guid}.txt")
        text = read_file_with_auto_encoding(text_file)
        text_inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        # 图像处理（读取对应的{guid}.img图片中的内容并转化为张量）
        image_file = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(image_file).convert("RGB")
        image_inputs = self.transform(image)
        # 标签处理（将train.txt文件中的情感转化为数字存储）
        label_map = {"positive": 0, "neutral": 1, "negative": 2}
        label = label_map[label] if label != "null" else -1
        return text_inputs, image_inputs, label

# 将文本数据、图像数据和标签合并成一个批次
def collate_fn(batch):
    text_inputs = {
        "input_ids": pad_sequence([item[0]["input_ids"].squeeze(0) for item in batch], batch_first=True, padding_value=0),
        "attention_mask": pad_sequence([item[0]["attention_mask"].squeeze(0) for item in batch], batch_first=True, padding_value=0)
    }
    image_inputs = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return text_inputs, image_inputs, labels

# 选择分词器模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 定义对图像数据进行的预处理操作（通过transform实现）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#调用上述函数，读取数据
train_data = load_data(TRAIN_FILE)
dataset = MultiModalDataset(train_data, DATA_PATH, tokenizer, transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#将数据集转化为模型能接受的格式
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 定义模型
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # 文本特征提取
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        # 图像特征提取
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)
        # 融合层
        self.fc = nn.Sequential(
            nn.Linear(768 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )

    #前向传播
    def forward(self, text_inputs, image_inputs):
        text_features = self.text_model(**text_inputs).pooler_output
        image_features = self.image_model(image_inputs)
        combined = torch.cat((text_features, image_features), dim=1)
        output = self.fc(combined)
        return output

# 训练
model = MultiModalModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for text_inputs, image_inputs, labels in train_loader:
        text_inputs = {key: val.squeeze(1).to(DEVICE) for key, val in text_inputs.items()}
        image_inputs = image_inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(text_inputs, image_inputs)
        loss = criterion(outputs, labels)
        #反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算训练准确率
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

# 验证
def evaluate(model, val_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for text_inputs, image_inputs, labels in val_loader:
            text_inputs = {key: val.squeeze(1).to(DEVICE) for key, val in text_inputs.items()}
            image_inputs = image_inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(text_inputs, image_inputs)
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

evaluate(model, val_loader)

# 测试（使用test_without_label.txt文件）
def predict_test(model, test_file, data_dir, tokenizer, transform):
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        next(f)  
        for line in f:
            guid, label = line.strip().split(",")
            test_data.append((guid, label))

    test_results = []
    model.eval()
    with torch.no_grad():
        for guid, _ in test_data:
            # 文本处理
            text_file = os.path.join(data_dir, f"{guid}.txt")
            if not os.path.exists(text_file):
                print(f"Warning: {text_file} not found, skipping...")
                continue
            text = read_file_with_auto_encoding(text_file)
            text_inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            text_inputs = {key: val.squeeze(1).to(DEVICE) for key, val in text_inputs.items()}

            # 图像处理
            image_file = os.path.join(data_dir, f"{guid}.jpg")
            if not os.path.exists(image_file):
                print(f"Warning: {image_file} not found, skipping...")
                continue
            image = Image.open(image_file).convert("RGB")
            image_inputs = transform(image).unsqueeze(0).to(DEVICE)

            # 模型预测
            outputs = model(text_inputs, image_inputs)
            prediction = torch.argmax(outputs, dim=1).item()
            label_map = {0: "positive", 1: "neutral", 2: "negative"}
            test_results.append((guid, label_map[prediction]))

    # 保存预测结果至test_predictions.txt文件
    output_file = os.path.join(os.path.dirname(test_file), "test_predictions.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("guid,tag\n")  
        for guid, label in test_results:
            f.write(f"{guid},{label}\n")
    print(f"Test predictions saved to {output_file}")

#获得预测结果
predict_test(model, TEST_FILE, DATA_PATH, tokenizer, transform)
