
# In[1]:
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

# 设置主题数量
NUM_TOPICS = 9

# 定义20种不同的符号
MARKERS = ['o', 's', '^', 'v', '<', '>', 'D', 'x', '+', '*', 'H', 'p', '8', '1', '2', '3', '4', 'P', 'd', '|']

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def safe_split(x):
    if isinstance(x, str):
        x = x.strip()
        return x.split() if x else []
    else:
        return []

def standardize_time_format(df):
    if '时间' not in df.columns:
        raise ValueError("DataFrame中缺少'时间'列")
    df['时间'] = df['时间'].astype(str)

    def format_date(date_str):
        for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
            try:
                date = pd.to_datetime(date_str, format=fmt, errors='coerce')
                if pd.notna(date):
                    return date.strftime(fmt)  # 使用原始格式
            except:
                continue
        return pd.NA

    df['时间'] = df['时间'].apply(format_date)
    return df

def train_lda_model(df, num_topics=NUM_TOPICS):
    texts = df['content_cutted'].apply(safe_split)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                          alpha=0.1 / num_topics, eta=0.01, minimum_probability=0.01,
                          update_every=1, chunksize=400, passes=20, random_state=1)
    return lda, dictionary

def calculate_topic_strength_by_time(df, lda, dictionary, num_topics=NUM_TOPICS):
    topic_strength_by_year = {}

    years = sorted(df['时间'].dropna().unique())

    for year in years:
        df_year = df[df['时间'] == year]
        texts = df_year['content_cutted'].apply(safe_split)
        corpus = [dictionary.doc2bow(text) for text in texts]

        if not corpus:
            print(f"No valid corpus data for year {year}")
            continue

        doc_topics = lda[corpus]
        topic_strength = np.zeros(num_topics)
        for doc_topic in doc_topics:
            for topic_id, prob in doc_topic:
                topic_strength[topic_id] += prob

        doc_count = len(doc_topics)
        if doc_count > 0:
            topic_strength /= doc_count

        topic_strength_by_year[year] = topic_strength
        print(f'Year: {year}, Topic Strength: {[float(f"{val:.4f}") for val in topic_strength]}')

    return topic_strength_by_year

def plot_topic_strength_over_time(topic_strength_by_year, num_topics=NUM_TOPICS):
    years = sorted(topic_strength_by_year.keys())
    plt.figure(figsize=(10, 6))
    for topic_id in range(num_topics):
        marker = MARKERS[topic_id % len(MARKERS)]
        topic_strength = [topic_strength_by_year[year][topic_id] for year in years]
        plt.plot(years, topic_strength, marker=marker, label=f'Topic {topic_id + 1}')
    plt.xlabel('年份')
    plt.ylabel('主题强度')
    plt.title('主题强度随时间演变')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('主题强度随时间演变.png')
    print("主题强度随时间演变.png已保存")

def plot_topic_strength_scatter(topic_strength):
    x = range(len(topic_strength))
    y = topic_strength.values
    labels = topic_strength.index

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='yellow', edgecolors='black', s=80)

    for i, val in enumerate(y):
        plt.text(x[i], val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(x, labels, rotation=0)
    plt.xlabel('主题编号')
    plt.ylabel('平均强度')
    plt.title('各主题的平均强度点状图')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('主题平均强度点状图.png')
    print("已生成主题平均强度点状图")

def save_topic_keywords_with_weights_to_excel_and_generate_wordclouds(lda, num_topics=NUM_TOPICS):
    data = []
    for topic_id in range(num_topics):
        keywords_with_weights = lda.show_topic(topic_id, topn=50)
        total_weight = sum(weight for _, weight in keywords_with_weights)
        normalized = [(kw, w / total_weight) for kw, w in keywords_with_weights]
        for kw, weight in normalized:
            data.append([topic_id + 1, kw, weight])
    df_keywords = pd.DataFrame(data, columns=['主题编号', '关键词', '权重'])
    df_keywords.to_excel('主题权重_.xlsx', index=False)
    print("包含关键词和权重的主题信息已保存到 '主题权重_.xlsx' 文件中")

    for topic_id in range(num_topics):
        topic_words = df_keywords[df_keywords['主题编号'] == topic_id + 1]
        word_freq = {row['关键词']: row['权重'] for _, row in topic_words.iterrows()}
        wc = WordCloud(width=800, height=600, background_color='white', font_path='./华文仿宋.ttf')
        plt.figure(figsize=(8, 6))
        plt.imshow(wc.generate_from_frequencies(word_freq), interpolation='bilinear')
        plt.title(f'主题 {topic_id + 1} 词云')
        plt.axis('off')
        plt.savefig(f"词云主题_{topic_id + 1}.png")
    print("每个主题的词云图已保存为 PNG 文件")

def visualize_lda_model(lda, corpus, dictionary):
    vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    print("LDA可视化已保存到 'lda_visualization.html' 文件中")

def plot_heatmap(topic_strength_by_year, num_topics=NUM_TOPICS):
    years = sorted(topic_strength_by_year.keys())
    heatmap_data = np.array([topic_strength_by_year[year] for year in years]).T
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=years,
                yticklabels=[f'Topic {i + 1}' for i in range(num_topics)])
    plt.title('主题强度随时间变化热图')
    plt.xlabel('时间')
    plt.ylabel('主题')
    plt.tight_layout()
    plt.savefig('主题强度热图.png')
    print("主题强度热图.png已保存")


def save_document_topic_distribution_with_max_topic(df, lda, dictionary, corpus):
    doc_topic_data = []
    for i, doc_bow in enumerate(corpus):
        topic_distribution = lda.get_document_topics(doc_bow, minimum_probability=0.01)
        topic_probs = [0] * NUM_TOPICS
        for topic_id, prob in topic_distribution:
            topic_probs[topic_id] = prob
        total_prob = sum(topic_probs)
        if total_prob > 0:
            topic_probs = [prob / total_prob for prob in topic_probs]
        max_topic_prob = max(topic_probs)
        max_topic = topic_probs.index(max_topic_prob)
        year = df['时间'].iloc[i]  # 获取当前文档对应的年份
        doc_topic_data.append([year, df['内容'].iloc[i]] + topic_probs + [max_topic + 1])
    columns = ['年份', '内容'] + [f'Topic {i + 1}' for i in range(NUM_TOPICS)] + ['最大主题编号']
    df_topic_distribution = pd.DataFrame(doc_topic_data, columns=columns)
    df_topic_distribution.to_excel('doc_topic_distribution.xlsx', index=False)
    print("文档主题分布已保存到 'doc_topic_distribution.xlsx' 文件中")


def export_topic_terms_distribution(lda, dictionary, num_topics=NUM_TOPICS):
    topic_terms_data = {}
    for topic_id in range(num_topics):
        term_list = lda.get_topic_terms(topic_id, topn=30)
        ids = [term[0] for term in term_list]
        probs = [term[1] for term in term_list]
        freqs = [dictionary.dfs.get(term_id, 0) for term_id in ids]
        topic_terms_data[f'主题{topic_id + 1}_词'] = [dictionary.id2token[term_id] for term_id in ids]
        topic_terms_data[f'主题{topic_id + 1}_P(w|z)'] = probs
        topic_terms_data[f'主题{topic_id + 1}_词频'] = freqs
    df_terms = pd.DataFrame(topic_terms_data)
    df_terms.to_excel('主题词分布.xlsx', index=False)
    print("主题词分布（P(w|z)）已保存到 '主题词分布.xlsx'")

# 主流程
if __name__ == "__main__":
    df = pd.read_excel('output.xlsx')
    df = standardize_time_format(df)
    df['content_cutted'] = df['content_cutted'].fillna('')

    lda, dictionary = train_lda_model(df, num_topics=NUM_TOPICS)
    corpus = [dictionary.doc2bow(safe_split(text)) for text in df['content_cutted']]

    topic_strength_by_year = calculate_topic_strength_by_time(df, lda, dictionary)
    topic_strength = pd.Series(np.mean(list(topic_strength_by_year.values()), axis=0),
                               index=[f'Topic {i+1}' for i in range(NUM_TOPICS)])
    plot_topic_strength_scatter(topic_strength)
    plot_topic_strength_over_time(topic_strength_by_year, num_topics=NUM_TOPICS)
    save_topic_keywords_with_weights_to_excel_and_generate_wordclouds(lda, num_topics=NUM_TOPICS)
    visualize_lda_model(lda, corpus, dictionary)
    plot_heatmap(topic_strength_by_year, num_topics=NUM_TOPICS)
    save_document_topic_distribution_with_max_topic(df, lda, dictionary, corpus)
    export_topic_terms_distribution(lda, dictionary, num_topics=NUM_TOPICS)
















