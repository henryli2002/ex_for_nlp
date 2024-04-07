import torch
from collections import Counter
import spacy

# 读取文件得到词列表
words_list = []
with open('lmtraining.txt', 'r', encoding='utf-8') as file:
    for line in file:
        words = line.strip().split()
        words_list += words

# 计算词频去掉过低频词和停用词
words_counts = Counter(words_list)
min_freq = 3
nlp = spacy.load('en_core_web_sm')
english_stopwords = nlp.Defaults.stop_words
filtered_words = [word for word in words_list if min_freq <= words_counts[word] and word not in english_stopwords]

# 去除重复词得到词库的字典
i = 0
words_dict = dict()
for word in filtered_words:
    if word not in words_dict:
        words_dict[word] = i
        i += 1

# 改用tensor类型的数据来加速计算
vocab_size = len(words_dict)
vocab_metrix = torch.zeros((vocab_size, vocab_size))



