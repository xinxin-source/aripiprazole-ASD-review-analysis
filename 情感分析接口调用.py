import json
import os
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
access_key_id = '替换为自己的'
access_key_secret = '替换为自己的'
# 创建AcsClient实例
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)
lst = []
sj = pd.read_excel('data.xlsx')
sj['情感分数'] = ''
for i in tqdm(range(len(sj))):
    try:
        keywords = sj['内容'].loc[i]
        request = CommonRequest()
        # domain和version是固定值
        request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
        request.set_version('2020-06-29')
        # action name可以在API文档里查到
        request.set_action_name('GetSaChGeneral')
        # 需要add哪些param可以在API文档里查到
        request.add_query_param('ServiceCode', 'alinlp')
        request.add_query_param('Text', keywords)
        request.add_query_param('TokenizerId', 'GENERAL_CHN')
        response = client.do_action_with_exception(request)
        resp_obj = json.loads(response)['Data']
        resp_obj2 = json.loads(resp_obj)['result']['positive_prob']
        sj['情感分数'].loc[i] = resp_obj2
        
        lst.append(resp_obj2)
    except:
        print('error')


sj.to_excel('百度情感分析.xlsx', index=False)
