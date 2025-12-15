import pandas as pd
import re

# 获取指定工作表中的数据
df = pd.read_excel('工作簿1.xlsx')

# 将文本列转换为字符串类型
df['文本'] = df['文本'].astype(str)

# 删除[]里面的内容
df['文本'] = df['文本'].apply(lambda x: re.sub(r'\[.*?\]', '', x))

# 去重
df = df.drop_duplicates()

# 过滤空值
df = df.dropna()

# 删除只含有@***或文本长度小于2字符的无效评论
df = df[~(df['文本'].str.match(r'^@\*\*\*$') | (df['文本'].str.len() < 2))]

# 去除非中文字符
df['文本'] = df['文本'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]', '', x))

# 将结果保存为 Excel 文件
output_path = '处理后.xlsx'
df.to_excel(output_path, index=False)