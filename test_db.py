from py2neo import Graph
import streamlit as st

def test_database():
    try:
        # 连接数据库
        client = Graph(uri="bolt://localhost:7687", auth=("neo4j", "11111111"))
        
        # 测试基本查询
        st.write("### 测试数据库连接")
        st.write("连接成功！")
        
        # 查询电影数量
        movie_count = client.run("MATCH (m:电影) RETURN count(m) as count").data()[0]['count']
        st.write(f"### 数据库中的电影数量: {movie_count}")
        
        # 查询示例电影及其所有属性
        st.write("### 示例电影数据（显示所有属性）")
        movies = client.run("MATCH (m:电影) RETURN m LIMIT 5").data()
        for movie in movies:
            st.write("电影节点数据:")
            st.json(movie['m'])
            
        # 测试特定查询（使用不同的查询方式）
        st.write("### 测试特定查询")
        test_movie = "肖申克的救赎"
        
        # 方式1：精确匹配
        result1 = client.run(f"MATCH (m:电影) WHERE m.名称 = '{test_movie}' RETURN m").data()
        st.write("精确匹配结果:")
        if result1:
            st.json(result1[0])
        else:
            st.write("未找到精确匹配")
            
        # 方式2：模糊匹配
        result2 = client.run(f"MATCH (m:电影) WHERE m.名称 CONTAINS '{test_movie}' RETURN m").data()
        st.write("模糊匹配结果:")
        if result2:
            st.json(result2[0])
        else:
            st.write("未找到模糊匹配")
            
        # 方式3：查看所有可能的名称格式
        result3 = client.run(f"MATCH (m:电影) WHERE m.名称 CONTAINS '肖申克' RETURN m").data()
        st.write("包含'肖申克'的电影:")
        if result3:
            for movie in result3:
                st.json(movie['m'])
        else:
            st.write("未找到相关电影")
            
    except Exception as e:
        st.error(f"数据库连接错误: {str(e)}")

if __name__ == "__main__":
    st.title("Neo4j 数据库测试")
    test_database() 