import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# 从文件读取词典
words_dict_path = './data/words_dict.txt'
words_dict = {}
with open(words_dict_path, 'r', encoding='utf-8') as file:
    for line in file:
        word, number = line.strip().split(':')
        words_dict[word] = int(number)

# 读取并分割数字列表（多行）
numerical_list_path = './data/numerical_list.txt'
numerical_list = []
with open(numerical_list_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 将当前行的数字添加到列表中,虽然这里只有一行其实
        numerical_list.extend([int(number) for number in line.strip().split('\t')])

# 用tensor类型的数据来加速计算
vocab_size = len(words_dict)
vocab_matrix = torch.zeros((vocab_size, vocab_size)).to(device)

# 制作共现矩阵,设窗口大小为K
K = 5
for i, word in enumerate(numerical_list):
    for j in range(max(i-K, 0), min(i+K, len(numerical_list))):
        if numerical_list[j] != word and numerical_list[j] != -1:
            vocab_matrix[word, numerical_list[j]] += 1

# SVD降维
U, S, V = torch.linalg.svd(vocab_matrix.float(), full_matrices=False)
k = int(100)  # 降维到N维
U_reduced = U[:, :k]
S_reduced = S[:k]
V_reduced = V[:, :k]

# 计算非0奇异值的数量
non_zero_singular_values = torch.sum(S > 0).item()

# 选取的奇异值之和
sum_selected_singular_values = torch.sum(S[:k]).item()

# 全部奇异值之和
sum_all_singular_values = torch.sum(S).item()

# 二者的比例
ratio = sum_selected_singular_values / sum_all_singular_values

print(f"非零奇异值的数量: {non_zero_singular_values}")
print(f"选取的前{k}个奇异值之和: {sum_selected_singular_values:.4f}")
print(f"全部奇异值之和: {sum_all_singular_values:.4f}")
print(f"选取的奇异值之和与全部奇异值之和的比例: {ratio:.4f}")

# 计算降维后的数据矩阵
vec_sta = torch.matmul(U_reduced, torch.diag(S_reduced))

# 给文件增加sim_svd
modified_lines = []
with open('./data/wordsim353_agreed.txt', 'r', encoding='utf-8') as file:
    for line in file:
        sim_svd = 0
        words = line.strip().split()
        if words[1] in words_dict and words[2] in words_dict:
            sim_svd = F.cosine_similarity(vec_sta[words_dict[words[1]]].unsqueeze(0), vec_sta[words_dict[words[2]]].unsqueeze(0)).cpu().numpy()
        modified_line = line.rstrip('\n') + f'\t{sim_svd}\n'
        modified_lines.append(modified_line) 
        
with open('./output/svd_output.txt', 'w', encoding='utf-8') as file:
    file.writelines(modified_lines)