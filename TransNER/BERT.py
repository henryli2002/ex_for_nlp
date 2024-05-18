from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm

# 图像处理部分
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义训练的参数
num_epochs = 100
batch_size = 32
learning_rate = 5e-5
max_length = 100
num_labels = 6

# 初始化berttokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-chinese",
    cache_dir=None,
    force_download=False,
)


# 训练函数
def train(model, data_loader, optimizer, device, loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-1)):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 忽略attention_mask为0的位置和labels为-1的位置
        active_loss = attention_mask.view(-1) == 1
        active_labels = labels.view(-1)
        active_loss = active_loss & (active_labels != -1) 
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = active_labels[active_loss]

        if len(active_labels) > 0:  # 仅当有有效的标签时才计算损失
            loss = loss_fn(active_logits, active_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_description(f"Step {step}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Average loss: {avg_loss:.4f}")


# 训练函数
def test(model, data_loader, device, idx_to_label, ignore_index=-1, save=0):
    model.eval()
    total_loss = 0
    true_labels = []
    predictions_list = []  
    sentences = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, batch in progress_bar:
            batch = [item.to(device) for item in batch]
            input_ids, attention_mask, labels = batch

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # 只计算有效的损失
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()

            # 忽略 ignore_index 的位置
            active_positions = labels != ignore_index
            true_labels.extend(labels[active_positions].cpu().numpy())
            predictions_list.extend(predictions[active_positions].cpu().numpy())
        
            for i in range(len(input_ids)):
                sentence_labels = predictions[i][attention_mask[i] == 1].cpu().numpy()
                sentences.append(" ".join(str(idx_to_label[label]) for label in sentence_labels))
    
    avg_loss = total_loss / len(data_loader)
    precision = precision_score(true_labels, predictions_list, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions_list, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions_list, average="weighted", zero_division=0)

    # 将结果写入文件
    with open('./output/2021213688.txt', "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(
        f"Test Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )

    return precision, true_labels, predictions_list, sentences


# encoder
def prepare_data(data_path, label_path, labels_dict):
    data = []
    labels = []

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = " ".join(line.strip().split())
            data.append(line)

    encoded_data = tokenizer.batch_encode_plus(
        data,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_token_type_ids=False,
        return_tensors="pt",
    )

    with open(label_path, "r", encoding="utf-8") as file:
        for line, encoding in zip(file, encoded_data['input_ids']):
            line_labels = line.strip().split()
            label_ids = [labels_dict.get(label, -1) for label in line_labels]

            # 处理特殊标记：为[CLS]标记添加-1
            label_ids = [-1] + label_ids

            # 为每个[SEP]标记添加-1
            sep_positions = (encoding == tokenizer.sep_token_id).nonzero()
            for pos in reversed(sep_positions):
                label_ids.insert(pos.item(), -1)

            # 截断和填充逻辑
            label_ids = label_ids[:max_length]
            label_ids += [-1] * (max_length - len(label_ids))

            labels.append(label_ids)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return encoded_data, labels_tensor


# 绘制图像
def plot_metrics(labels, predictions, num_classes=8):
    # 绘制混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f"./pic/Confusion_Matrix_{num_classes}.png")
    plt.show()

    # 绘制标签和预测的分布柱状图
    labels_df = pd.DataFrame({"Labels": labels, "Category": "True"})
    predictions_df = pd.DataFrame({"Labels": predictions, "Category": "Predicted"})
    combined_df = pd.concat([labels_df, predictions_df])

    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_df, x="Labels", hue="Category")
    plt.title("Distribution of True Labels vs Predictions")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(f"./pic/Distribution_{num_classes}.png")
    plt.show()


# 主函数
def main():
    # 数据准备
    labels_set = set()
    with open("data/train_TAG.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split()
            labels_set.update(line)
    labels_dict = {label: idx for idx, label in enumerate(sorted(labels_set))}
    print(labels_dict)
    train_path = "data/train.txt"
    train_TAG_path = "data/train_TAG.txt"
    train_data, train_labels = prepare_data(train_path, train_TAG_path, labels_dict)

    # 打印输入数据的键和形状
    print("Encoded data shapes:")
    for key, value in train_data.items():
        print(f"{key}: {value.shape}")

    # 打印标签数据的形状
    print("Labels shape:", train_labels.shape)

    dataset = TensorDataset(
            train_data['input_ids'], 
            train_data['attention_mask'], 
            train_labels
        )
    
    # 准备训练参数，每轮从总数据中抽2000个数据训练
    dataset_size = len(dataset)
    subset_size = int(0.01 * dataset_size)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    subset_indices = indices[:subset_size]
    sampler = SubsetRandomSampler(subset_indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    


    # 模型、优化器和损失函数的准备
    model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(labels_dict))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练过程
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model=model, data_loader=data_loader, optimizer=optimizer, device=device)
        # 保存和加载模型，此处需要手动改模型路径
        torch.save(model.state_dict(), f"./models/model_{epoch}e.pth")

    # 验证过程
    best_ac = 0  # 记录最高值
    best_epoch = 0  # 记录最好epoch
    accuracy_per_epoch = []
    for epoch in range(num_epochs):
        ac = 0
        model.load_state_dict(torch.load(f"./models/model_{epoch}e.pth"))
        dev_path = "data/dev.txt"
        dev_TAG_path = "data/dev_TAG.txt"
        dev_data, dev_labels = prepare_data(dev_path, dev_TAG_path, labels_dict)

        dev_dataset = TensorDataset(
            dev_data["input_ids"],
            dev_data["attention_mask"],
            dev_labels,
        )
        dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size)
        ac, true_labels, predictions_list = test(model, dev_data_loader, device)
        accuracy_per_epoch.append(ac)
        if ac > best_ac:
            best_ac = ac
            best_epoch = epoch
        print(f"best_epoch:{best_epoch}")

    # 绘制验证图像
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_per_epoch, marker='o', linestyle='-', color='b')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(range(num_epochs), [f"Epoch {i+1}" for i in range(num_epochs)]) 
    plt.savefig("./pic/accuracy_over_epochs.png")
    plt.show()

    # 测试模型性能
    model.load_state_dict(torch.load(f"./models/model_{best_epoch}e.pth"))
    model.load_state_dict(torch.load(f"./models/model_{38}e.pth"))
    test_path = "data/test.txt"
    test_TAG_path = "data/test_TAG.txt"
    test_data, test_labels = prepare_data(test_path, test_TAG_path, labels_dict)

    test_dataset = TensorDataset(
        test_data["input_ids"],
        test_data["attention_mask"],
        test_labels,
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    idx_to_label = {v: k for k, v in labels_dict.items()}
    ac, true_labels, predictions_list, sentences = test(model, test_data_loader, device, idx_to_label, save=1)

    # 绘制混淆矩阵和数据分布
    plot_metrics(true_labels, predictions_list, num_classes=len(labels_dict)) 


if __name__ == "__main__":
    main()
