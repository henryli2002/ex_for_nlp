from transformers import BertTokenizer, BertForSequenceClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 图像处理部分
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 定义训练的参数
num_epochs = 10
batch_size = 64
learning_rate = 5e-5
max_length = 100
num_labels = 6

tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir=None,
        force_download=False,
    )

# 训练函数
def train(model, data_loader, optimizer, device, loss_fn=None):
    """训练模型的函数
    参数:
    model: 要训练的模型
    data_loader: 数据加载器
    optimizer: 优化器
    device: 设备（CPU或CUDA）
    loss_fn: 损失函数（可选，如果模型内部已定义，则不需要，目前还没有实现）
    """
    model.train()
    total_loss = 0.
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        batch = [item.to(device) for item in batch]  # 将数据移动到指定设备
        input_ids, token_type_ids, attention_mask, labels = batch  # 解包数据

        optimizer.zero_grad()  # 清空梯度
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
        loss = outputs.loss  # 获取损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Average loss: {avg_loss:.4f}")


# 测试函数
def test(model, data_loader, device, dev=False, bi_class=False):
    """测试模型的函数
    参数:
    model: 要测试的模型
    data_loader: 数据加载器
    device: 设备（CPU或CUDA）
    dev: 是否是验证模式
    bi_class: 是否转化为2分类验证
    """
    model.eval()
    total_loss = 0
    labels_list = []
    results_list = []  # 用于收集预测结果
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, batch in progress_bar:
            batch = [item.to(device) for item in batch]
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            if bi_class:
                 # 将标签映射为二分类问题
                binary_labels = (labels >= 3).to(torch.int64)
                binary_predictions = (predictions >= 3).to(torch.int64)

                labels_list.extend(binary_labels.cpu().numpy())
                results_list.extend(binary_predictions.cpu().numpy())
            else:
                labels_list.extend(labels.cpu().numpy())
                results_list.extend(predictions.cpu().numpy())
    if bi_class:
        avg_loss = total_loss / len(data_loader)
        accuracy = (np.array(labels_list) == np.array(results_list)).mean()
        f1 = f1_score(labels_list, results_list, average='binary')  # 计算加权F1分数
        recall = recall_score(labels_list, results_list, average='binary')  # 计算加权召回率
    else:
        avg_loss = total_loss / len(data_loader)
        accuracy = (np.array(labels_list) == np.array(results_list)).mean()
        f1 = f1_score(labels_list, results_list, average='weighted')  # 计算加权F1分数
        recall = recall_score(labels_list, results_list, average='weighted')  # 计算加权召回率
        
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}")

    if dev == True:
        return accuracy
    return labels_list, results_list


# 绘制图像
def plot_metrics(labels, predictions, num_classes=6):
    # 绘制混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'./pic/Confusion_Matrix_{num_classes}.png')
    plt.show()

    # 绘制标签和预测的分布柱状图
    labels_df = pd.DataFrame({'Labels': labels, 'Category': 'True'})
    predictions_df = pd.DataFrame({'Labels': predictions, 'Category': 'Predicted'})
    combined_df = pd.concat([labels_df, predictions_df])

    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined_df, x='Labels', hue='Category')
    plt.title('Distribution of True Labels vs Predictions')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig(f'./pic/Distribution_{num_classes}.png')
    plt.show()

 # 数据编码


# encoder
def encoder(df):
    """对文本数据进行编码的函数
    参数:
    df: 包含文本数据的DataFrame
    """
    train_data_encoded = tokenizer.batch_encode_plus(
        list(zip(df['text1'].values.tolist(), df['text2'].values.tolist())),
        add_special_tokens=True,
        truncation=True,
        padding='max_length', 
        max_length=max_length,
        return_tensors='pt'
    )
    train_labels = df['labels'].values.tolist()
    return train_data_encoded, train_labels


# 主函数
def main():
    # 数据准备
    column_names = ['text1', 'text2', 'labels']
    df_train = pd.read_csv('Chinese-STS-B/sts-b-train.txt', sep='\t', names=column_names)
    df_dev = pd.read_csv('Chinese-STS-B/sts-b-dev.txt', sep='\t', names=column_names)
    df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t', names=column_names)


    # 模型、优化器和损失函数的准备
    train_data, train_labels = encoder(df_train)
    input_ids = train_data['input_ids']
    token_type_ids = train_data['token_type_ids']
    attention_mask = train_data['attention_mask']
    train_labels = torch.Tensor(train_labels)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练过程
    model.to(device)
    for epoch in range(num_epochs):  
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model=model, data_loader=data_loader, optimizer=optimizer, device=device)
        # 保存和加载模型，此处需要手动改模型路径
        torch.save(model.state_dict(), f'./models/model_{epoch}e.pth')
    
   
    # 验证过程
    best_ac = 0  # 记录最高值
    best_epoch = 0  # 记录最好epoch
    for epoch in range(num_epochs):
        ac = 0
        model.load_state_dict(torch.load(f'./models/model_{epoch}e.pth')) 
        dev_data, dev_labels = encoder(df_dev)
        dev_dataset = TensorDataset(dev_data['input_ids'], dev_data['token_type_ids'], dev_data['attention_mask'], torch.Tensor(dev_labels))
        dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size)
        ac = test(model, dev_data_loader, device,dev=True)
        if ac > best_ac:
            best_ac = ac
            best_epoch = epoch
        print(f"best_epoch:{best_epoch}")


    
 



if __name__ == "__main__":
    main()