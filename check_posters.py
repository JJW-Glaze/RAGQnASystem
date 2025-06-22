import py2neo
import os

def check_poster_paths():
    # 连接Neo4j数据库
    client = py2neo.Graph('bolt://localhost:7687', user='neo4j', password='11111111')
    
    # 查询有海报的电影数量
    count_query = '''
    MATCH (m:电影)
    WHERE m.海报路径 <> ''
    RETURN COUNT(m) as count
    '''
    count_result = client.run(count_query).data()
    print(f'\n有海报的电影数量: {count_result[0]["count"]}')
    
    # 查询示例电影及其海报路径
    sample_query = '''
    MATCH (m:电影)
    WHERE m.海报路径 <> ''
    RETURN m.名称 as 电影名称, m.海报路径 as 海报路径
    LIMIT 5
    '''
    print('\n示例电影及其海报路径:')
    for record in client.run(sample_query).data():
        print(f'电影: {record["电影名称"]}')
        print(f'海报路径: {record["海报路径"]}')
        # 检查海报文件是否实际存在
        poster_path = os.path.join('data', 'posters', record["海报路径"])
        print(f'海报文件存在: {os.path.exists(poster_path)}\n')

if __name__ == '__main__':
    check_poster_paths() 