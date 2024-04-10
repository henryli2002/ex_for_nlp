from collections import Counter
import spacy

# 读取文件得到词列表
words_list = []
with open('./data/lmtraining.txt', 'r', encoding='utf-8') as file:
    for line in file:
        words = line.strip().split()
        words_list += words

# 去掉停用词
nlp = spacy.load('en_core_web_sm')
english_stopwords = nlp.Defaults.stop_words
filtered_stopwords = [word for word in words_list if word not in english_stopwords]

# 计算词频去掉过低频词
words_counts = Counter(filtered_stopwords)
min_freq = 10
filtered_words = [word for word in filtered_stopwords if min_freq <= words_counts[word]]

# 去除重复词得到词库的字典
i = 0
words_dict = dict()
for word in filtered_words:
    if word not in words_dict:
        words_dict[word] = i
        i += 1

# 将词转换为相应的数字
numerical_list = [words_dict[word] for word in filtered_words]

# 保存过滤后的词列表到文件
with open('./data/numerical_list.txt', 'w', encoding='utf-8') as file:
    for number in numerical_list:
        file.write(f'{number}\t') 

# 保存words_dict到文件
with open('./data/words_dict.txt', 'w', encoding='utf-8') as file:
    for word, number in words_dict.items():
        file.write(f'{word}:{number}\n')


# 用于生成SGNS的训练集
def generate_training_data(words_list, context_window):
    for center in range(len(words_list)):
        for context in range(-context_window, context_window + 1):
            context_index = center + context
            if context_index < 0 or context_index >= len(words_list) or center == context_index:
                continue
            yield words_list[center], words_list[context_index]

K = 2
training_data = list(generate_training_data(words_list=numerical_list, context_window=K))

# 保存training_data到文件
with open('./data/sgns_training_data.txt', 'w', encoding='utf-8') as file:
    for input_label in training_data:
        file.write(f'{input_label}\n')