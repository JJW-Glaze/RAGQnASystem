import random
import torch
from torch import nn
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer
from tqdm import tqdm
from seqeval.metrics import f1_score
import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cache_model = 'best_roberta_rnn_model_ent_aug'

def get_data(path,max_len=None):
    all_text,all_tag = [],[]
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen,tag = [],[]
    for data in all_data:
        data = data.split(' ')
        if(len(data)!=2):
            if len(sen)>2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te,ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text,all_tag

class rule_find:
    def __init__(self):
        self.idx2type = ["电影","导演","演员","编剧","类型","国家","语言","评分","年份","评价人数"]
        self.type2idx = {type: idx for idx, type in enumerate(self.idx2type)}
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.type2idx))]

        for type in self.idx2type:
            with open(os.path.join('data','ent_aug',f'{type}.txt'),encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en)>=2:
                    self.ahos[self.type2idx[type]].add_word(en,en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self,sen):
        rule_result = []
        mp = {}
        all_res = []
        all_ty = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for j in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i,res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


#找出tag(label)中的所有实体及其下表，为实体动态替换/随机掩码策略/实体动态拼接做准备
def find_entities(tag):
    result = []#[(2,3,'药品'),(7,10,'药品商')]
    label_len = len(tag)
    i = 0
    while(i<label_len):
        if(tag[i][0]=='B'):
            type = tag[i].strip('B-')
            j=i+1
            while(j<label_len and tag[j][0]=='I'):
                j += 1
            result.append((i,j-1,type))
            i=j
        else:
            i = i + 1
    return result


class tfidf_alignment():
    def __init__(self):
        eneities_path = os.path.join('data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        self.tag_2_entity = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [ent for ent in entities if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model
    def align(self,ent_list):
        new_result = {}
        for s,e,cls,ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]

            if max_score >= 0.5:
                new_result[cls]= self.tag_2_entity[cls][max_idx]
        return new_result


class Entity_Extend:
    def __init__(self):
        self.idx2type = ["电影","导演","演员","编剧","类型","国家","语言","评分","年份","评价人数"]
        self.type2idx = {type: idx for idx, type in enumerate(self.idx2type)}
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.type2idx))]
        
        # 加载实体词典
        for type in self.idx2type:
            with open(os.path.join('data','ent_aug',f'{type}.txt'),encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en)>=2:
                    self.ahos[self.type2idx[type]].add_word(en,en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()
            
    def find_entity(self,text):
        entities = []
        for i in range(len(self.ahos)):
            for end_index, original_value in self.ahos[i].iter(text):
                start_index = end_index - len(original_value) + 1
                entities.append((start_index,end_index+1,self.idx2type[i]))
        return entities
        
    def extend_entity(self,text,entities):
        # 对每个实体进行扩展
        new_entities = []
        for start,end,type in entities:
            # 获取实体文本
            entity_text = text[start:end]
            
            # 根据实体类型进行不同的扩展策略
            if type == "电影":
                # 电影名称扩展：添加"电影"、"影片"等后缀，以及"的"前缀
                suffixes = ["电影", "影片", "片"]
                prefixes = ["的"]
                for suffix in suffixes:
                    new_entity = entity_text + suffix
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
                for prefix in prefixes:
                    new_entity = prefix + entity_text
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
            
            elif type in ["导演","演员","编剧"]:
                # 人名扩展：添加"先生"、"女士"等后缀，以及"的"前缀
                suffixes = ["先生", "女士", "老师", "导演", "演员", "编剧"]
                prefixes = ["的"]
                for suffix in suffixes:
                    new_entity = entity_text + suffix
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
                for prefix in prefixes:
                    new_entity = prefix + entity_text
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
            
            elif type == "类型":
                # 电影类型扩展：添加"类型"、"题材"等后缀，以及"的"前缀
                suffixes = ["类型", "题材", "风格", "片"]
                prefixes = ["的"]
                for suffix in suffixes:
                    new_entity = entity_text + suffix
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
                for prefix in prefixes:
                    new_entity = prefix + entity_text
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
            
            elif type in ["国家","语言"]:
                # 国家和语言扩展：添加"的"等连接词
                prefixes = ["来自", "的", "在"]
                for prefix in prefixes:
                    new_entity = prefix + entity_text
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
            
            elif type in ["评分","年份","评价人数"]:
                # 数值类实体扩展：添加单位或说明
                if type == "评分":
                    suffixes = ["分", "评分", "星"]
                elif type == "年份":
                    suffixes = ["年", "年代", "上映"]
                else:
                    suffixes = ["人", "人次", "评价"]
                
                for suffix in suffixes:
                    new_entity = entity_text + suffix
                    if new_entity in text and (text.index(new_entity),text.index(new_entity)+len(new_entity),type) not in entities:
                        new_entities.append((text.index(new_entity),text.index(new_entity)+len(new_entity),type))
        
        return entities + new_entities

    def entities_extend(self, text, label_list, ents):
        # 根据实体列表更新标签
        for start, end, type in ents:
            if start < len(text) and end <= len(text):
                for i in range(start, end):
                    if i == start:
                        label_list[i] = f'B-{type}'
                    else:
                        label_list[i] = f'I-{type}'
        return text, label_list




class Nerdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizer,max_len,tag2idx,is_dev=False,enhance_data=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len= max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
        self.entity_extend = Entity_Extend()
        self.enhance_data = enhance_data
    def __getitem__(self, x):
        text = ''.join(self.all_text[x])  # 将字符列表转换为字符串
        label = self.all_label[x]
        
        ents = self.entity_extend.find_entity(text)
        extended_entities = self.entity_extend.extend_entity(text, ents)
        
        # 根据扩展后的实体更新标签
        label_list = ['O'] * len(text)  # 初始化所有标签为 'O'
        for start, end, type in extended_entities:
            if start < len(text) and end <= len(text):  # 确保实体边界在文本范围内
                for i in range(start, end):
                    if i == start:
                        label_list[i] = f'B-{type}'
                    else:
                        label_list[i] = f'I-{type}'
        
        if self.is_dev:
            max_len = min(len(text)+2,500)
        else:
            # 几种策略
            if self.enhance_data and e>=7 and e%2==1:
                ents = find_entities(label_list)
                text,label_list = self.entity_extend.entities_extend(text,label_list,ents)
            max_len = self.max_len
        text = text[:max_len - 2]
        label_list = label_list[:max_len - 2]

        x_len = len(text)
        assert len(text)==len(label_list), f"Text length ({len(text)}) != Label length ({len(label_list)})"
        text_idx = self.tokenizer.encode(text, add_special_tokens=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label_list] + [self.tag2idx['<PAD>']]

        text_idx +=[0]*(max_len-len(text_idx))
        label_idx +=[self.tag2idx['<PAD>']]*(max_len-len(label_idx))
        return torch.tensor(text_idx),torch.tensor(label_idx),x_len
    def __len__(self):
        return len(self.all_text)




def build_tag2idx(all_tag):
    tag2idx = {'<PAD>':0, 'O':1}
    # 添加所有可能的标签
    entity_types = ["电影","导演","演员","编剧","类型","国家","语言","评分","年份","评价人数"]
    for type in entity_types:
        tag2idx[f'B-{type}'] = len(tag2idx)
        tag2idx[f'I-{type}'] = len(tag2idx)
    return tag2idx




class Bert_Model(nn.Module):
    def __init__(self,model_name,hidden_size,tag_num,bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(input_size=768,hidden_size=hidden_size,num_layers=2,batch_first=True,bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_size*2,tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self,x,label=None):
        bert_0,_ = self.bert(x,attention_mask=(x>0),return_dict=False)
        gru_0,_ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1).squeeze(0)

def merge(model_result_word,rule_result):
    result = model_result_word+rule_result
    result = sorted(result,key=lambda x:len(x[-1]),reverse=True)
    check_result = []
    mp = {}
    for res in result:
        if res[0] in mp or res[1] in mp:
            continue
        check_result.append(res)
        for i in range(res[0],res[1]+1):
            mp[i] = 1
    return check_result

def get_ner_result(model,tokenizer,sen,rule,tfidf_r,device,idx2tag):
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)

    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        word = sen[res[0]:res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)

    merge_result = merge(model_result_word, rule_result)
    # print('模型结果',model_result_word)
    # print('规则结果',rule_result)
    tfidf_result = tfidf_r.align(merge_result)
    #print('整合结果', merge_result)
    #print('tfidf对齐结果', tfidf_result)
    return tfidf_result

if __name__ == "__main__":
    all_text,all_label = get_data(os.path.join('data','ner_data_aug.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size = 0.02, random_state = 42)

    #加载太慢了，预处理一下
    if os.path.exists('tmp_data/tag2idx.npy'):
        with open('tmp_data/tag2idx.npy','rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('tmp_data/tag2idx.npy','wb') as f:
            pickle.dump(tag2idx,f)

    idx2tag = list(tag2idx)

    max_len = 30
    epoch = 30
    batch_size = 32
    hidden_size = 64
    bi = True
    model_name='model/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lr =1e-5
    is_train=True

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    train_dataset = Nerdataset(train_text,train_label,tokenizer,max_len,tag2idx,enhance_data=True)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx,is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name,hidden_size,len(tag2idx),bi)
    if os.path.exists(f'model/best_roberta_rnn_model_ent_aug.pt'):
        model.load_state_dict(torch.load('model/best_roberta_rnn_model_ent_aug.pt'))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    bestf1 = -1
    
    # 创建日志文件
    log_file = 'training_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('训练配置信息：\n')
        f.write(f'训练集大小: {len(train_dataset)}\n')
        f.write(f'验证集大小: {len(dev_dataset)}\n')
        f.write(f'批次大小: {batch_size}\n')
        f.write(f'学习率: {lr}\n')
        f.write(f'最大序列长度: {max_len}\n')
        f.write(f'隐藏层大小: {hidden_size}\n')
        f.write(f'双向GRU: {bi}\n')
        f.write('-' * 50 + '\n')
    
    if is_train:
        print("\n开始训练...")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(dev_dataset)}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {lr}")
        print(f"最大序列长度: {max_len}")
        print(f"隐藏层大小: {hidden_size}")
        print(f"双向GRU: {bi}")
        print("-" * 50)
        
        for e in range(epoch):
            model.train()
            loss_sum = 0
            ba = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {e+1}/{epoch}')
            for x,y,batch_len in progress_bar:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = model(x,y)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                ba += 1
                progress_bar.set_postfix({'loss': f'{loss_sum/ba:.4f}'})
            
            model.eval()
            all_pre = []
            all_label = []
            with torch.no_grad():
                for x,y,batch_len in tqdm(dev_dataloader, desc='验证中'):
                    x = x.to(device)
                    pre = model(x)
                    pre = [idx2tag[i] for i in pre[1:batch_len+1]]
                    all_pre.append(pre)
                    label = [idx2tag[i] for i in y[0][1:batch_len+1]]
                    all_label.append(label)
            
            f1 = f1_score(all_pre, all_label)
            
            # 记录训练日志
            with open(log_file, 'a', encoding='utf-8') as f:
                if f1 > bestf1:
                    bestf1 = f1
                    log_msg = f'Epoch {e+1}/{epoch} - Loss: {loss_sum/ba:.4f} - F1: {f1:.4f} (最佳模型已保存)\n'
                    f.write(log_msg)
                    print(f'\n{log_msg.strip()}')
                    torch.save(model.state_dict(),f'model/{cache_model}.pt')
                else:
                    log_msg = f'Epoch {e+1}/{epoch} - Loss: {loss_sum/ba:.4f} - F1: {f1:.4f}\n'
                    f.write(log_msg)
                    print(f'\n{log_msg.strip()}')

    rule = rule_find()
    tfidf_r = tfidf_alignment()

    while(True):
        sen = input('请输入:')
        print(get_ner_result(model, tokenizer, sen, rule, tfidf_r,device,idx2tag))
