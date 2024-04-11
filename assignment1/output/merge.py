# 仅用来把数据变成标准格式，你愿意的话可以一开始就弄成标准的

lines = []
with open('./output/svd_output.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.replace("[", "").replace("]", "")
        parts = line.strip().split('\t')
        sim = float(parts[-1])
        formatted_sim = f"{sim:.4f}"
        parts[-1] = formatted_sim
        modified_line = '\t'.join(parts) + '\n'
        lines.append(modified_line)


modified_lines = []  
with open('./output/sgns_output.txt', 'r', encoding='utf-8') as file:
    for line, words in zip(lines, file):
        sim = words.strip().split('\t')[-1]
        sim = float(sim)
        modified_line = line.rstrip('\n') + f'\t{sim:.4f}\n'
        modified_lines.append(modified_line)
    
with open('output/2021213688.txt', 'w', encoding='utf-8') as file:
    file.writelines(modified_lines)