import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import json
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 确保图片目录存在
os.makedirs('img/performance', exist_ok=True)

class PerformanceVisualizer:
    def __init__(self, data_file='performance_data.json'):
        """初始化可视化器，加载性能数据"""
        self.data = self.load_data(data_file)
        self.module_metrics = self.data.get('module_metrics', {})
        
    def load_data(self, data_file):
        """从JSON文件加载性能数据"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换时间戳字符串为datetime对象
            for key in ['resource_usage', 'cache_hit_rates', 'success_rates', 'throughput']:
                if 'timestamps' in data[key]:
                    data[key]['timestamps'] = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') 
                                             if isinstance(ts, str) else ts
                                             for ts in data[key]['timestamps']]
            
            return data
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return {}
    
    def plot_response_times(self):
        """绘制响应时间箱线图"""
        response_times = self.data['response_times']
        
        plt.figure(figsize=(10, 6))
        box_data = [response_times[module] for module in ['意图识别', '实体识别', '知识图谱查询', '推荐生成', '整体响应']]
        
        box = plt.boxplot(box_data, patch_artist=True, labels=['意图识别', '实体识别', '知识图谱查询', '推荐生成', '整体响应'])
        
        # 设置箱体颜色
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('各模块响应时间分布')
        plt.ylabel('响应时间 (毫秒)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标注
        for i, module in enumerate(['意图识别', '实体识别', '知识图谱查询', '推荐生成', '整体响应']):
            median_val = np.median(response_times[module])
            plt.text(i+1, median_val, f'{median_val:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('img/performance/response_times_boxplot.png', dpi=300)
        plt.close()
    
    def plot_resource_usage(self):
        """绘制系统资源使用趋势图"""
        resource_data = self.data['resource_usage']
        timestamps = resource_data['timestamps']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # CPU和内存使用率
        ax1.plot(timestamps, resource_data['cpu'], 'b-', label='CPU使用率')
        ax1.plot(timestamps, resource_data['memory'], 'r-', label='内存使用率')
        ax1.set_ylabel('使用率 (%)')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('系统资源使用趋势')
        
        # GPU使用率和内存
        if 'gpu_util' in resource_data and 'gpu_memory' in resource_data:
            ax2.plot(timestamps, resource_data['gpu_util'], 'g-', label='GPU使用率')
            ax2.plot(timestamps, resource_data['gpu_memory'], 'm-', label='GPU内存使用率')
            ax2.set_ylabel('使用率 (%)')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 格式化x轴日期
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.set_xlabel('时间')
        
        plt.tight_layout()
        plt.savefig('img/performance/resource_usage_trend.png', dpi=300)
        plt.close()
    
    def plot_cache_hit_rates(self):
        """绘制缓存命中率趋势图"""
        cache_data = self.data['cache_hit_rates']
        timestamps = cache_data['timestamps']
        
        plt.figure(figsize=(10, 6))
        
        for module in ['意图识别', '实体识别', '知识图谱查询', '推荐生成']:
            plt.plot(timestamps, cache_data[module], label=module)
        
        plt.plot(timestamps, cache_data['整体缓存'], 'k-', linewidth=2, label='整体平均')
        
        plt.title('各模块缓存命中率趋势')
        plt.xlabel('时间')
        plt.ylabel('缓存命中率')
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        plt.savefig('img/performance/cache_hit_rates_trend.png', dpi=300)
        plt.close()
    
    def plot_success_rates(self):
        """绘制各模块成功率趋势图"""
        success_data = self.data['success_rates']
        timestamps = success_data['timestamps']
        
        plt.figure(figsize=(10, 6))
        
        for module in ['意图识别', '实体识别', '知识图谱查询', '推荐生成']:
            plt.plot(timestamps, success_data[module], label=module)
        
        plt.title('各模块请求成功率趋势')
        plt.xlabel('时间')
        plt.ylabel('成功率')
        plt.ylim(0.8, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        plt.savefig('img/performance/success_rates_trend.png', dpi=300)
        plt.close()
    
    def plot_throughput(self):
        """绘制系统吞吐量（QPS）趋势图"""
        throughput_data = self.data['throughput']
        timestamps = throughput_data['timestamps']
        qps = throughput_data['qps']
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(timestamps, qps, 'b-')
        plt.fill_between(timestamps, 0, qps, alpha=0.3, color='blue')
        
        plt.title('系统吞吐量趋势')
        plt.xlabel('时间')
        plt.ylabel('每秒查询数 (QPS)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 标注最大和最小值
        max_qps = max(qps)
        min_qps = min(qps)
        max_idx = qps.index(max_qps)
        min_idx = qps.index(min_qps)
        
        plt.annotate(f'最大值: {max_qps:.1f}', xy=(timestamps[max_idx], max_qps),
                    xytext=(10, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.annotate(f'最小值: {min_qps:.1f}', xy=(timestamps[min_idx], min_qps),
                    xytext=(10, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        plt.savefig('img/performance/throughput_trend.png', dpi=300)
        plt.close()
    
    def plot_intent_metrics(self):
        """绘制意图识别性能指标"""
        if '意图识别' not in self.module_metrics:
            return
        
        metrics = self.module_metrics['意图识别']
        
        plt.figure(figsize=(8, 5))
        
        x = np.arange(len(metrics['accuracy']))
        width = 0.35
        
        plt.bar(x - width/2, metrics['accuracy'], width, label='准确率')
        plt.bar(x + width/2, metrics['f1'], width, label='F1值')
        
        plt.title('意图识别性能指标')
        plt.xlabel('测试样本')
        plt.ylabel('分数')
        plt.ylim(0.8, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 添加均值线
        plt.axhline(y=np.mean(metrics['accuracy']), color='b', linestyle='--', alpha=0.7)
        plt.axhline(y=np.mean(metrics['f1']), color='orange', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('img/performance/intent_recognition_metrics.png', dpi=300)
        plt.close()
    
    def plot_entity_metrics(self):
        """绘制实体识别性能指标"""
        if '实体识别' not in self.module_metrics:
            return
        
        metrics = self.module_metrics['实体识别']
        
        plt.figure(figsize=(8, 5))
        
        # 准备数据
        categories = ['精确率', '召回率', 'F1值']
        values = [np.mean(metrics['precision']), np.mean(metrics['recall']), np.mean(metrics['f1'])]
        std_dev = [np.std(metrics['precision']), np.std(metrics['recall']), np.std(metrics['f1'])]
        
        x = np.arange(len(categories))
        
        # 绘制条形图带误差线
        plt.bar(x, values, yerr=std_dev, align='center', alpha=0.7, ecolor='black', capsize=10)
        
        # 添加数值标签
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.title('实体识别性能指标')
        plt.xticks(x, categories)
        plt.ylabel('分数')
        plt.ylim(0.8, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('img/performance/entity_recognition_metrics.png', dpi=300)
        plt.close()
    
    def plot_recommendation_metrics(self):
        """绘制推荐系统性能指标"""
        if '推荐系统' not in self.module_metrics:
            return
        
        metrics = self.module_metrics['推荐系统']
        
        plt.figure(figsize=(8, 5))
        
        categories = ['Precision@k', 'Recall@k', 'NDCG']
        values = [np.mean(metrics['precision@k']), np.mean(metrics['recall@k']), np.mean(metrics['ndcg'])]
        std_dev = [np.std(metrics['precision@k']), np.std(metrics['recall@k']), np.std(metrics['ndcg'])]
        
        x = np.arange(len(categories))
        
        # 绘制条形图带误差线
        plt.bar(x, values, yerr=std_dev, align='center', alpha=0.7, ecolor='black', capsize=10)
        
        # 添加数值标签
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.title('推荐系统性能指标')
        plt.xticks(x, categories)
        plt.ylabel('分数')
        plt.ylim(0.7, 0.9)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('img/performance/recommendation_metrics.png', dpi=300)
        plt.close()
    
    def plot_knowledge_graph_metrics(self):
        """绘制知识图谱性能指标"""
        if '知识图谱' not in self.module_metrics:
            return
        
        metrics = self.module_metrics['知识图谱']
        
        # 创建复合图表：条形图和折线图
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics['query_complexity']))
        width = 0.35
        ax1.bar(x - width/2, metrics['query_complexity'], width, label='查询复杂度', color='skyblue')
        ax1.set_xlabel('测试样本')
        ax1.set_ylabel('查询复杂度', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(x, metrics['path_length'], 'r-', label='路径长度')
        ax2.set_ylabel('平均路径长度', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('知识图谱查询指标')
        plt.grid(False)
        
        plt.tight_layout()
        plt.savefig('img/performance/knowledge_graph_metrics.png', dpi=300)
        plt.close()
    
    def plot_performance_overview(self):
        """绘制系统性能总览雷达图"""
        categories = ['响应速度', '成功率', '缓存命中率', '系统负载', '吞吐量']
        
        # 计算平均指标
        avg_response_time = np.mean(self.data['response_times']['整体响应'])
        max_response_time = 500  # 假设最大可接受响应时间为500ms
        response_score = max(0, 1 - avg_response_time / max_response_time)
        
        avg_success_rate = np.mean([np.mean(self.data['success_rates'][module]) 
                                  for module in ['意图识别', '实体识别', '知识图谱查询', '推荐生成']])
        
        avg_cache_hit = np.mean(self.data['cache_hit_rates']['整体缓存'])
        
        # 系统负载（CPU和内存使用率的均值）
        avg_cpu = np.mean(self.data['resource_usage']['cpu'])
        avg_memory = np.mean(self.data['resource_usage']['memory'])
        system_load_score = 1 - ((avg_cpu + avg_memory) / 2) / 100  # 负载越低越好
        
        avg_qps = np.mean(self.data['throughput']['qps'])
        max_qps = 100  # 假设最大期望QPS为100
        throughput_score = min(1, avg_qps / max_qps)
        
        values = [response_score, avg_success_rate, avg_cache_hit, system_load_score, throughput_score]
        
        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]
        categories += categories[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        
        ax.set_ylim(0, 1)
        ax.set_title('系统性能总览')
        
        # 添加数值标签
        for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('img/performance/performance_overview_radar.png', dpi=300)
        plt.close()
    
    def generate_all_plots(self):
        """生成所有性能图表"""
        self.plot_response_times()
        self.plot_resource_usage()
        self.plot_cache_hit_rates()
        self.plot_success_rates()
        self.plot_throughput()
        self.plot_intent_metrics()
        self.plot_entity_metrics()
        self.plot_recommendation_metrics()
        self.plot_knowledge_graph_metrics()
        self.plot_performance_overview()
        
        print("所有性能图表已生成到 img/performance/ 目录")

if __name__ == "__main__":
    # 加载并可视化性能数据
    visualizer = PerformanceVisualizer()
    visualizer.generate_all_plots() 