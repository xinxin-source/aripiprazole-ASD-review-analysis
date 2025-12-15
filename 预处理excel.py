
# In[1]:
import pandas as pd
import re
import jieba as jb
import jieba.posseg as psg
from collections import Counter, defaultdict

# 定义停用词和同义词加载函数
def stopwordslist(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)  # 使用集合提高查找效率
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        stopwords = set()
    return stopwords

def load_synonyms(filepath):
    synonyms = defaultdict(list)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split('=')
                if len(words) > 1:
                    main_word = words[0]
                    synonym_words = words[1].split(',')
                    for word in synonym_words:
                        synonyms[word].append(main_word)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    return synonyms

# 加载用户词典、停用词和同义词
jb.load_userdict('dict.txt')
stop_list = stopwordslist('stopwords.txt')
synonyms = load_synonyms('tongyici.txt')

def chinese_word_cut(mytext):
    jb.initialize()
    flag_list = ['n', 'v', 'a', 'vn', 'ns', 'nr', 'nt']  # 扩展词性标签，增加地名(ns)、人名(nr)、机构团体(nt)
    word_list = []

    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
        if word and word not in stop_list and seg_word.flag in flag_list:
            # 同义词替换
            if word in synonyms:
                word = synonyms[word][0]
            word_list.append(word)

    # 合并短词为一个词
    word_list = [word for word in word_list if len(word) > 1]

    return " ".join(word_list)

# 处理文本数据
def process_text(text):
    return chinese_word_cut(text)

# 从 Excel 文件读取数据
df = pd.read_excel('data.xlsx')
comments = df['内容'].astype(str)

# 分批处理数据
batch_size = 1000
num_batches = (len(comments) + batch_size - 1) // batch_size

with open('cutted.txt', 'w', encoding='utf-8') as f:
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(comments))
        batch_comments = comments[start_idx:end_idx]

        for comment in batch_comments:
            line_seg = process_text(comment)
            if line_seg:
                f.write(line_seg + '\n')

# 读取预处理后的文本文件并更新 DataFrame
with open('cutted.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

df['content_cutted'] = pd.Series([line.strip() for line in lines]).reindex(df.index, fill_value='')

# 输出到 Excel 文件
df.to_excel('output.xlsx', index=False)

# 统计词频
word_counts = Counter()
for line in lines:
    words = line.strip().split()
    word_counts.update(words)

# 按照词频从高到低排序
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# 将排序后的词频结果写入文件
with open('词频统计.txt', 'w', encoding='utf-8') as f:
    for word, count in sorted_word_counts:
        f.write(f"{word}\t{count}\n")

print("分词处理和词频统计已完成。")



# In[4]:


# In[5]:
# 定义生成词云图函数


# In[6]:


# In[7]:

