

# In[1]:
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')  # 使用 TkAgg 后端
    import matplotlib.pyplot as plt
    import gensim
    from gensim import corpora, models
    from gensim.models import CoherenceModel
    import matplotlib.font_manager as fm

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

    # 计算一致性评分
    def coherence(num_topics):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=50)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        print("Coherence Score:", coherence_score)
        return coherence_score

    # 绘制一致性评分折线图
    font_path = "./华文仿宋.ttf"  # 字体文件路径
    font_prop = fm.FontProperties(fname=font_path, size=7) # 指定字体属性
    x = range(1, 11)  # 主题范围数量
    y = [coherence(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目', fontproperties=font_prop)  # 设置字体属性
    plt.ylabel('一致性大小', fontproperties=font_prop)  # 设置字体属性
    plt.title('主题-一致性变化情况', fontproperties=font_prop)  # 设置字体属性
    #plt.show()
    plt.savefig('coherence_plot.png')  # 保存图形为PNG文件
