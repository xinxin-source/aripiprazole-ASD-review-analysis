
# In[1]:
import pandas as pd
import networkx as nx
from pyvis.network import Network

# 读取Excel文件
file_path = '主题权重_.xlsx'
df = pd.read_excel(file_path)

# 确保 '权重' 列为数值类型，并将NaN值替换为0
df['权重'] = pd.to_numeric(df['权重'], errors='coerce').fillna(0)

# 创建一个空的NetworkX图形对象
G = nx.Graph()

# 记录词的颜色和词属于哪些主题
word_colors = {}
word_topics = {}

# 定义一组不同的主题颜色
topic_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# 遍历每一行，创建节点和边
for _, row in df.iterrows():
    topic_id = f"主题 {int(row['主题编号'])}"  # 强制转换为字符串
    word = row['关键词']
    weight = row['权重']

    # 获取颜色
    color = topic_colors[int(row['主题编号']) % len(topic_colors)]  # 防止越界

    # 添加主题节点
    if not G.has_node(topic_id):
        G.add_node(topic_id, label=topic_id, color=color, size=60)

    # 记录词属于的主题
    if word not in word_topics:
        word_topics[word] = set()
    word_topics[word].add(topic_id)

    # 根据权重调整词节点大小
    word_size = 5 + weight * 300  # 增加大小变化范围，使差异更明显

    # 添加词节点
    if not G.has_node(word):
        G.add_node(word, label=word, color=color, size=word_size)
        word_colors[word] = color  # 初始颜色为主题颜色

    # 根据权重调整边的粗细
    edge_width = weight * 250

    # 添加边
    if not G.has_edge(topic_id, word):
        G.add_edge(topic_id, word, weight=weight, title=f'权重: {weight:.4f}', width=edge_width)

# 更新词的颜色，确保只有属于多个主题的词用红色
for word, topics in word_topics.items():
    if len(topics) > 1:
        word_colors[word] = 'red'
        G.nodes[word]['color'] = 'red'
    else:
        # 如果词只属于一个主题，确保其颜色为主题颜色
        single_topic = list(topics)[0]
        topic_index = int(single_topic.split()[1])  # 获取主题索引
        word_colors[word] = topic_colors[topic_index]
        G.nodes[word]['color'] = topic_colors[topic_index]

# 使用PyVis创建交互式网络图
net = Network(notebook=True, width="1000px", height="700px")
net.from_nx(G)

# 配置节点颜色和大小
for node in net.nodes:
    node['title'] = node['label']  # 显示标签
    node['value'] = node['size']  # 使用大小作为值
    node['color'] = word_colors.get(node['label'], node['color'])  # 设置颜色

# 配置边的粗细
for edge in net.edges:
    edge['width'] = edge['width']

# 获取网络图的HTML内容
net_html = net.generate_html()

# 创建带图例的HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>主题词关系图</title>
    <style>
        body {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            flex-direction: column;
        }}
        .legend {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .legend div {{
            margin: 0 10px;
            display: inline-block;
        }}
        .network {{
            width: 1000px;
            height: 700px;
        }}
    </style>
</head>
<body>
    <div class="legend">
        <div><span style="color:red;">■</span> 多个主题词</div>
"""

# 添加每个主题的图例项
for i, color in enumerate(topic_colors[:df['主题编号'].max() + 1]):
    html_content += f'<div><span style="color:{color};">■</span> 主题 {i}</div>'

html_content += f"""
    </div>
    <div class="network">
        {net_html}
    </div>
</body>
</html>
"""

# 将带图例的HTML写入文件
with open('主题词关系图.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

