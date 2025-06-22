import os
import re
import py2neo
from tqdm import tqdm
import argparse
import pandas as pd
from datetime import datetime
import numpy as np


def safe_split(value, delimiter='/'):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(delimiter)]
    return []

def clean_rating(value):
    if pd.isna(value):
        return ""
    return str(float(value))

def clean_count(value):
    if pd.isna(value):
        return ""
    return str(int(value))

def clean_percentage(value):
    if pd.isna(value):
        return ""
    return value.replace('%', '')

def clean_filename(name):
    """
    清理文件名，处理特殊字符
    """
    # 移除特殊字符
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    # 移除空格
    name = name.strip()
    return name

#导入普通实体
def import_entity(client, type, entity):
    def create_node(client, type, name):
        order = """create (n:%s{名称:"%s"})"""%(type,name)
        client.run(order)

    print(f'正在导入{type}类数据')
    for en in tqdm(entity):
        create_node(client, type, en)

def get_poster_path(movie_name):
    """
    根据电影名称查找对应的海报文件
    返回相对路径（如果找到）或空字符串（如果未找到）
    """
    # 获取原始中文名（第一个空格前的部分）
    original_name = movie_name.split()[0]
    # 清理电影名称
    clean_name = clean_filename(original_name)
    
    # 可能的海报文件扩展名
    extensions = ['.jpg', '.jpeg', '.png']
    
    # 尝试直接匹配
    for ext in extensions:
        poster_path = os.path.join("data", "posters", f"{clean_name}{ext}")
        if os.path.exists(poster_path):
            return f"{clean_name}{ext}"
    
    # 如果直接匹配失败，尝试更灵活的匹配
    posters_dir = os.path.join("data", "posters")
    if os.path.exists(posters_dir):
        # 获取所有海报文件
        poster_files = os.listdir(posters_dir)
        
        # 1. 尝试移除所有特殊字符后的完全匹配
        for filename in poster_files:
            name_without_ext = os.path.splitext(filename)[0]
            if clean_filename(name_without_ext) == clean_name:
                return filename
        
        # 2. 尝试部分匹配
        for filename in poster_files:
            name_without_ext = os.path.splitext(filename)[0]
            clean_file_name = clean_filename(name_without_ext)
            # 如果清理后的文件名包含电影名称，或电影名称包含文件名
            if clean_name in clean_file_name or clean_file_name in clean_name:
                return filename
        
        # 3. 尝试更模糊的匹配（移除所有标点符号后的匹配）
        clean_name_no_punct = re.sub(r'[^\w\s]', '', clean_name)
        for filename in poster_files:
            name_without_ext = os.path.splitext(filename)[0]
            clean_file_name_no_punct = re.sub(r'[^\w\s]', '', name_without_ext)
            if clean_name_no_punct in clean_file_name_no_punct or clean_file_name_no_punct in clean_name_no_punct:
                return filename
    
    return ""

#导入电影类实体
def import_movie_data(client, type, entity):
    print(f'正在导入{type}类数据')
    for movie in tqdm(entity):
        # 获取海报路径
        poster_path = get_poster_path(movie["电影名称"])
        
        node = py2neo.Node(type,
                          名称=movie["电影名称"],
                          导演=movie["导演"] if pd.notna(movie["导演"]) else "",
                          编剧=movie["编剧"] if pd.notna(movie["编剧"]) else "",
                          主演=movie["主演"] if pd.notna(movie["主演"]) else "",
                          类型=movie["类型"] if pd.notna(movie["类型"]) else "",
                          制片国家=movie["制片国家/地区"] if pd.notna(movie["制片国家/地区"]) else "",
                          语言=movie["语言"] if pd.notna(movie["语言"]) else "",
                          上映日期=movie["上映日期"] if pd.notna(movie["上映日期"]) else "",
                          片长=movie["片长"] if pd.notna(movie["片长"]) else "",
                          又名=movie["又名"] if pd.notna(movie["又名"]) else "",
                          IMDb=movie["IMDb"] if pd.notna(movie["IMDb"]) else "",
                          豆瓣评分=clean_rating(movie["豆瓣评分"]),
                          评价人数=clean_count(movie["评价人数"]),
                          五星比例=clean_percentage(movie["五星比例"]),
                          四星比例=clean_percentage(movie["四星比例"]),
                          三星比例=clean_percentage(movie["三星比例"]),
                          二星比例=clean_percentage(movie["二星比例"]),
                          一星比例=clean_percentage(movie["一星比例"]),
                          剧情简介=movie["剧情简介"] if pd.notna(movie["剧情简介"]) else "",
                          海报路径=poster_path
                          )
        client.create(node)

def create_all_relationship(client, all_relationship):
    def create_relationship(client, type1, name1, relation, type2, name2):
        order = """match (a:%s{名称:"%s"}),(b:%s{名称:"%s"}) create (a)-[r:%s]->(b)"""%(type1,name1,type2,name2,relation)
        client.run(order)
    print("正在导入关系.....")
    for type1, name1, relation, type2, name2 in tqdm(all_relationship):
        create_relationship(client, type1, name1, relation, type2, name2)

def create_movie_node(tx, movie_data):
    query = """
    CREATE (m:电影 {
        名称: $name,
        豆瓣评分: $rating,
        上映日期: $release_date,
        剧情简介: $description,
        片长: $duration,
        制片国家: $country,
        语言: $language,
        评价人数: $rating_count,
        海报路径: $poster_path
    })
    """
    tx.run(query, 
           name=movie_data['name'],
           rating=movie_data['rating'],
           release_date=movie_data['release_date'],
           description=movie_data['description'],
           duration=movie_data['duration'],
           country=movie_data['country'],
           language=movie_data['language'],
           rating_count=movie_data['rating_count'],
           poster_path=movie_data['poster_path'])

if __name__ == "__main__":
    #连接数据库的一些参数
    parser = argparse.ArgumentParser(description="通过douban_movies.csv文件,创建一个电影知识图谱")
    parser.add_argument('--website', type=str, default='bolt://localhost:7687', help='neo4j的连接网站')
    parser.add_argument('--user', type=str, default='neo4j', help='neo4j的用户名')
    parser.add_argument('--password', type=str, default='11111111', help='neo4j的密码')
    parser.add_argument('--dbname', type=str, default='neo4j', help='数据库名称')
    args = parser.parse_args()

    #连接...
    client = py2neo.Graph(args.website, user=args.user, password=args.password, name=args.dbname)

    #将数据库中的内容删光
    is_delete = input('注意:是否删除neo4j上的所有实体 (y/n):')
    if is_delete=='y':
        client.run("match (n) detach delete (n)")

    #读取电影数据
    df = pd.read_csv('./data/douban_movies.csv')
    
    # 确保posters目录存在
    posters_dir = os.path.join("data", "posters")
    if not os.path.exists(posters_dir):
        os.makedirs(posters_dir)
        print(f"创建海报目录: {posters_dir}")
    else:
        print(f"海报目录已存在: {posters_dir}")
        # 统计海报文件数量
        poster_files = [f for f in os.listdir(posters_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"当前海报文件数量: {len(poster_files)}")
    
    #所有实体
    all_entity = {
        "电影": [],
        "导演": [],
        "演员": [],
        "编剧": [],
        "类型": [],
        "国家": [],
        "语言": [],
        "评分": [],
        "年份": [],
        "评价人数": [],
        "IMDb": []
    }
    
    # 实体间的关系
    relationship = []
    
    for _, row in df.iterrows():
        # 添加电影实体
        movie_name = row["电影名称"]
        all_entity["电影"].append({
            "电影名称": movie_name,
            "导演": row["导演"] if pd.notna(row["导演"]) else "",
            "编剧": row["编剧"] if pd.notna(row["编剧"]) else "",
            "主演": row["主演"] if pd.notna(row["主演"]) else "",
            "类型": row["类型"] if pd.notna(row["类型"]) else "",
            "制片国家/地区": row["制片国家/地区"] if pd.notna(row["制片国家/地区"]) else "",
            "语言": row["语言"] if pd.notna(row["语言"]) else "",
            "上映日期": row["上映日期"] if pd.notna(row["上映日期"]) else "",
            "片长": row["片长"] if pd.notna(row["片长"]) else "",
            "又名": row["又名"] if pd.notna(row["又名"]) else "",
            "IMDb": row["IMDb"] if pd.notna(row["IMDb"]) else "",
            "豆瓣评分": row["豆瓣评分"] if pd.notna(row["豆瓣评分"]) else "",
            "评价人数": row["评价人数"] if pd.notna(row["评价人数"]) else "",
            "五星比例": row["五星比例"] if pd.notna(row["五星比例"]) else "",
            "四星比例": row["四星比例"] if pd.notna(row["四星比例"]) else "",
            "三星比例": row["三星比例"] if pd.notna(row["三星比例"]) else "",
            "二星比例": row["二星比例"] if pd.notna(row["二星比例"]) else "",
            "一星比例": row["一星比例"] if pd.notna(row["一星比例"]) else "",
            "剧情简介": row["剧情简介"] if pd.notna(row["剧情简介"]) else ""
        })
        
        # 处理导演
        directors = safe_split(row["导演"])
        all_entity["导演"].extend(directors)
        relationship.extend([("导演", director, "执导", "电影", movie_name) for director in directors])
        
        # 处理演员
        actors = safe_split(row["主演"])
        all_entity["演员"].extend(actors)
        relationship.extend([("演员", actor, "出演", "电影", movie_name) for actor in actors])
        
        # 处理编剧
        writers = safe_split(row["编剧"])
        all_entity["编剧"].extend(writers)
        relationship.extend([("编剧", writer, "编剧", "电影", movie_name) for writer in writers])
        
        # 处理类型
        genres = safe_split(row["类型"])
        all_entity["类型"].extend(genres)
        relationship.extend([("电影", movie_name, "属于", "类型", genre) for genre in genres])
        
        # 处理国家
        countries = safe_split(row["制片国家/地区"])
        all_entity["国家"].extend(countries)
        relationship.extend([("电影", movie_name, "制作于", "国家", country) for country in countries])
        
        # 处理语言
        languages = safe_split(row["语言"])
        all_entity["语言"].extend(languages)
        relationship.extend([("电影", movie_name, "使用语言", "语言", language) for language in languages])
        
        # 处理评分
        if pd.notna(row["豆瓣评分"]):
            rating = clean_rating(row["豆瓣评分"])
            all_entity["评分"].append(rating)
            relationship.append(("电影", movie_name, "评分", "评分", rating))
        
        # 处理评价人数
        if pd.notna(row["评价人数"]):
            rating_count = clean_count(row["评价人数"])
            all_entity["评价人数"].append(rating_count)
            relationship.append(("电影", movie_name, "评价人数", "评价人数", rating_count))
        
        # 处理上映年份
        if pd.notna(row["上映日期"]):
            release_date = row["上映日期"]
            if isinstance(release_date, str):
                year = re.search(r'\d{4}', release_date)
                if year:
                    year = year.group()
                    all_entity["年份"].append(year)
                    relationship.append(("电影", movie_name, "上映年份", "年份", year))
        
        # 处理IMDb
        if pd.notna(row["IMDb"]):
            imdb_id = row["IMDb"]
            all_entity["IMDb"].append(imdb_id)
            relationship.append(("电影", movie_name, "IMDb链接", "IMDb", imdb_id))

    # 去重
    relationship = list(set(relationship))
    all_entity = {k: list(set(v)) if k != "电影" else v for k, v in all_entity.items()}
    
    # 保存关系
    with open("./data/rel_aug.txt", 'w', encoding='utf-8') as f:
        for rel in relationship:
            f.write(" ".join(rel))
            f.write('\n')

    # 保存实体
    if not os.path.exists('data/ent_aug'):
        os.mkdir('data/ent_aug')
    for k, v in all_entity.items():
        with open(f'data/ent_aug/{k}.txt', 'w', encoding='utf8') as f:
            if k != '电影':
                for i, ent in enumerate(v):
                    f.write(ent + ('\n' if i != len(v)-1 else ''))
            else:
                for i, ent in enumerate(v):
                    f.write(ent['电影名称'] + ('\n' if i != len(v)-1 else ''))

    # 将属性和实体导入到neo4j
    for k in all_entity:
        if k != "电影":
            import_entity(client, k, all_entity[k])
        else:
            import_movie_data(client, k, all_entity[k])
    create_all_relationship(client, relationship)