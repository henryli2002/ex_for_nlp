import torch
from collections import Counter
import spacy

device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取文件得到词列表
words_list = []
with open('lmtraining.txt', 'r', encoding='utf-8') as file:
    for line in file:
        words = line.strip().split()
        words_list += words

# 去掉停用词
filtered_words_list = [word for word in words_list if word not in english_stopwords]

# 计算词频去掉过低频词
words_counts = Counter(filtered_words_list)
min_freq = 3
nlp = spacy.load('en_core_web_sm')
english_stopwords = nlp.Defaults.stop_words
filtered_words = [word for word in filtered_words_list if min_freq <= words_counts[word]]

# 去除重复词得到词库的字典
i = 0
words_dict = dict()
for word in filtered_words:
    if word not in words_dict:
        words_dict[word] = i
        i += 1

# 用tensor类型的数据来加速计算
vocab_size = len(words_dict)
vocab_matrix = torch.zeros((vocab_size, vocab_size)).float().to(device)

# 制作共现矩阵,设窗口大小为K
K = 5
for i, word in enumerate(words_list):
    if word in words_dict:
        for j in range(max(i-K, 0), min(i+K, len(words_list))):
            if words_list[j] in words_dict and words_list[j] != word:
                vocab_matrix[words_dict[word], words_dict[words_list[j]]] += 1

# 打印共现矩阵
print(vocab_matrix)

U, S, V = torch.linalg.svd(vocab_matrix, full_matrices=False)
k = int(vocab_matrix.shape[0] * 0.5)  # 降维到N维
U_reduced = U[:, :k]
S_reduced = S[:k]
V_reduced = V[:, :k]

# 计算降维后的数据矩阵
vec_sta = torch.matmul(U_reduced, torch.diag(S_reduced))


# 给文件增加sim_svd
modified_lines = []
with open('wordsim353_agreed.txt', 'r', encoding='utf-8') as file:
    for line in file:
        sim_svd = 0
        words = line.strip().split()
        if words[1] in words_dict and words[2] in words_dict:
            sim_svd = torch.nn.functional.cosine_similarity(vec_sta[words_dict[words[1]]], vec_sta[words_dict[words[2]]])
        
        
        modified_line = line.rstrip('\n') + f'\t{sim_svd}\n'
        modified_lines.append(modified_line) 
        