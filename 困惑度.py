

import matplotlib
matplotlib.use('TkAgg')  # 指定使用 TkAgg 后端
import matplotlib.pyplot as plt
import gensim
from gensim import corpora, models
import matplotlib.font_manager as fm
import numpy as np

# 设置随机种子
np.random.seed(42)
gensim.models.ldamodel.LdaModel.random_state = 42

# 准备数据
PATH = "cutted.txt"  # 已经进行了分词的文档

file_object2 = open(PATH, encoding='utf-8', errors='ignore').read().split('\n')
data_set = []  # 建立存储分词的列表
for i in range(len(file_object2)):
    result = []
    seg_list = file_object2[i].split()  # 读取每一行文本
    for w in seg_list:  # 读取每一行分词
        result.append(w)
    data_set.append(result)

dictionary = corpora.Dictionary(data_set)  # 构建 document-term matrix
corpus = [dictionary.doc2bow(text) for text in data_set]

# 计算困惑度
def perplexity(num_topics):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100, random_state=42)
    print(ldamodel.print_topics(num_topics=2, num_words=20))
    perplexity_score = abs(ldamodel.log_perplexity(corpus))  # 取困惑度的绝对值
    print("Perplexity:", perplexity_score)
    return perplexity_score


font_path = "./华文仿宋.ttf"  # 字体文件路径
font_prop = fm.FontProperties(fname=font_path, size=16) # 指定字体属性
x = range(1,11)  # 主题范围数量
y = [perplexity(i) for i in x]
plt.plot(x, y)
plt.xlabel('主题数目', fontproperties=font_prop)  # 设置字体属性
plt.ylabel('困惑度大小', fontproperties=font_prop)  # 设置字体属性
plt.title('主题-困惑度变化情况', fontproperties=font_prop)  # 设置字体属性
plt.savefig('perplexity_plot.png')  # 保存图形为PNG文件
#plt.show()

