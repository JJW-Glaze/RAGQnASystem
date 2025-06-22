import os
import random
#自动机 用于字符串匹配
import ahocorasick
import re
from tqdm import tqdm
import sys
import pandas as pd

class Build_Ner_data():
    """
    这是一个ner数据生成类。主要作用是将data文件夹下的douban_movies.csv文件中的文本打上标签。
    这里有十类标签["电影","导演","演员","编剧","类型","国家","语言","评分","年份","评价人数"]，
    每种标签所对应的实体在data文件夹下的f'{type}.txt'中
    这里将每种实体导入到ahocorasick中，对每个文本进行模式匹配。
    """
    def __init__(self):
        self.idx2type = ["电影", "导演", "演员", "编剧", "类型", "国家", "语言", "评分", "年份", "评价人数"]
        self.type2idx = {type: idx for idx, type in enumerate(self.idx2type)}
        self.max_len = 30
        self.p = ['，', '。', '！', '；', '：', ',', '.', '?', '!', ';']
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.idx2type))]

        for type in self.idx2type:
            with open(os.path.join('data', 'ent_aug', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                if len(en) >= 2:
                    self.ahos[self.type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def split_text(self, text):
        """
        将长文本随机分割为短文本

        :param arg1: 长文本
        :return: 返回一个list,代表分割后的短文本
        :rtype: list
        """
        text = text.replace('\n', ',')
        pattern = r'([，。！；：,.?!;])(?=.)|[？,]'

        sentences = []

        for s in re.split(pattern, text):
            if s and len(s) > 0:
                sentences.append(s)

        sentences_text = [x for x in sentences if x not in self.p]
        sentences_Punctuation = [x for x in sentences[1::2] if x in self.p]
        split_text = []
        now_text = ''

        # 随机长度,有15%的概率生成短文本 10%的概率生成长文本
        for i in range(len(sentences_text)):
            if (len(now_text) > self.max_len and random.random() < 0.9 or random.random() < 0.15) and len(now_text) > 0:
                split_text.append(now_text)
                now_text = sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text += sentences_Punctuation[i]
            else:
                now_text += sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text += sentences_Punctuation[i]
        if len(now_text) > 0:
            split_text.append(now_text)

        # 随机选取30%的数据,把末尾标点改为。
        for i in range(len(split_text)):
            if random.random() < 0.3:
                if split_text[i][-1] in self.p:
                    split_text[i] = split_text[i][:-1] + '。'
                else:
                    split_text[i] = split_text[i] + '。'
        return split_text

    def make_text_label(self, text):
        """
        通过ahocorasick类对文本进行识别，创造出文本的ner标签

        :param arg1: 文本
        :return: 返回一个list,代表标签
        :rtype: list
        """
        label = ['O'] * len(text)
        flag = 0
        mp = {}
        for type in self.idx2type:
            li = list(self.ahos[self.type2idx[type]].iter(text))
            if len(li) == 0:
                continue
            li = sorted(li, key=lambda x: len(x[1]), reverse=True)
            for en in li:
                ed, name = en
                st = ed - len(name) + 1
                if st in mp or ed in mp:
                    continue
                label[st:ed + 1] = ['B-' + type] + ['I-' + type] * (ed - st)
                flag = flag + 1
                for i in range(st, ed + 1):
                    mp[i] = 1
        return label, flag

def build_file(all_text, all_label):
    with open(os.path.join('data', 'ner_data_aug.txt'), "w", encoding="utf-8") as f:
        for text, label in zip(all_text, all_label):
            for t, l in zip(text, label):
                f.write(f'{t} {l}\n')
            f.write('\n')

if __name__ == "__main__":
    # 读取电影数据
    df = pd.read_csv('./data/douban_movies.csv')
    build_ner_data = Build_Ner_data()

    all_text, all_label = [], []

    # 处理电影数据中的文本字段
    text_fields = ["剧情简介", "电影名称", "导演", "编剧", "主演", "类型", "制片国家/地区", "语言"]
    
    for _, row in df.iterrows():
        for field in text_fields:
            if pd.notna(row[field]):
                text = str(row[field])
                # 对于包含多个值的字段（如导演、主演等），需要分别处理每个值
                if '/' in text:
                    values = [v.strip() for v in text.split('/')]
                    for value in values:
                        if len(value) > 0:
                            label, flag = build_ner_data.make_text_label(value)
                            if flag >= 1:
                                assert (len(value) == len(label))
                                all_text.append(value)
                                all_label.append(label)
                else:
                    label, flag = build_ner_data.make_text_label(text)
                    if flag >= 1:
                        assert (len(text) == len(label))
                        all_text.append(text)
                        all_label.append(label)

    build_file(all_text, all_label)

