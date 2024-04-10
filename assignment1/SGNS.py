import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        scores = self.linear(embeds)
        log_probs = F.log_softmax(scores, dim=1)
        return log_probs

# 从文件读取词典
words_dict_path = './data/words_dict.txt'
words_dict = {}
with open(words_dict_path, 'r', encoding='utf-8') as file:
    for line in file:
        word, number = line.strip().split(':')
        words_dict[word] = int(number)

training_data = []
with open('./data/sgns_training_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 移除换行符并去掉两侧的括号
        stripped_line = line.strip()[1:-1]
        # 分割输入词和上下文词
        input_word, context_word = stripped_line.split(', ')
        # 将词转换为整数
        training_data.append((int(input_word), int(context_word)))

# training_data已经是包含(input_word, context_word)对的列表
input_words = [pair[0] for pair in training_data]
context_words = [pair[1] for pair in training_data]

# 转换为张量
input_tensor = torch.tensor(input_words, dtype=torch.long)
context_tensor = torch.tensor(context_words, dtype=torch.long)

# 创建TensorDataset和DataLoader
dataset = TensorDataset(input_tensor, context_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

vocab_size, embedding_dim = len(words_dict), 1000
model = SkipGramModel(vocab_size, embedding_dim).to(device)  # 模型实例
loss_function = nn.NLLLoss()  # 损失函数
num_epochs = 10
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for input_batch, context_batch in dataloader:
        # 移动到设备
        input_batch = input_batch.to(device)
        context_batch = context_batch.to(device)

        # 重置梯度
        optimizer.zero_grad()

        # 前向传播
        log_probs = model(input_batch)

        # 计算损失
        loss = loss_function(log_probs, context_batch)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss}")


# 保存模型
torch.save(model.state_dict(), './model_skipgram.pth')

# 初始化一个相同结构的模型
model_to_use= SkipGramModel(vocab_size, embedding_dim).to(device)

# 加载保存的状态字典
model_to_use.load_state_dict(torch.load('./model_skipgram.pth'))

# 调整成评估模式
model_to_use.eval()

import torch
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保模型在正确的设备上
model_to_use.to(device)

# 从文件中收集词对的索引
word_pairs = []
lines = []
with open('./data/wordsim353_agreed.txt', 'r', encoding='utf-8') as file:
    for line in file:
        words = line.strip().split()
        lines.append(line.strip())  # 保存原始行以便之后使用
        if words[1] in words_dict and words[2] in words_dict:
            word_pairs.append((words_dict[words[1]], words_dict[words[2]]))

# 将词对索引转换为Tensor
word_idx_1, word_idx_2 = zip(*word_pairs)
word_idx_1 = torch.tensor(word_idx_1, dtype=torch.long, device=device)
word_idx_2 = torch.tensor(word_idx_2, dtype=torch.long, device=device)

# 计算嵌入
with torch.no_grad():
    embeddings_1 = model_to_use.embeddings(word_idx_1)
    embeddings_2 = model_to_use.embeddings(word_idx_2)
    # 计算余弦相似度
    similarities = F.cosine_similarity(embeddings_1, embeddings_2).cpu().numpy()

# 创建相似度字典，键为(词1, 词2)对，值为相似度
sim_dict = {pair: sim for pair, sim in zip(word_pairs, similarities)}

# 写入
with open('./output/sgns_output.txt', 'w', encoding='utf-8') as file:
    for line in lines:
        words = line.split()
        # 如果当前行的词对在sim_dict中，则获取其相似度；否则，相似度为0
        sim_sgns = sim_dict.get((words_dict.get(words[1], -1), words_dict.get(words[2], -1)), 0)
        modified_line = f'{line}\t{sim_sgns}\n'
        file.write(modified_line)

