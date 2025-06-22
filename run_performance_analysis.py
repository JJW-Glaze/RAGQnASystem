import os
import sys
import argparse
import logging
from performance_evaluation import PerformanceMonitor, generate_sample_data
from performance_visualization import PerformanceVisualizer
from training_performance import TrainingPerformanceAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RAG系统性能评估与可视化')
    parser.add_argument('--action', choices=['generate', 'visualize', 'analyze_training', 'all'], default='all',
                        help='执行操作：generate=生成数据, visualize=可视化, analyze_training=分析训练性能, all=全部执行')
    parser.add_argument('--data_file', default='performance_data.json', 
                        help='性能数据文件路径')
    parser.add_argument('--output_dir', default='img/performance',
                        help='输出图像目录')
    parser.add_argument('--training_logs', action='store_true',
                        help='是否分析训练日志')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据选择的操作执行
    if args.action in ['generate', 'all']:
        logger.info("生成性能评估数据...")
        monitor = generate_sample_data()
        monitor.save_performance_data(args.data_file)
        logger.info(f"性能数据已保存到 {args.data_file}")
    
    if args.action in ['visualize', 'all']:
        logger.info("生成性能可视化图表...")
        visualizer = PerformanceVisualizer(args.data_file)
        visualizer.generate_all_plots()
        logger.info(f"性能图表已保存到 {args.output_dir} 目录")
    
    if args.action in ['analyze_training', 'all'] or args.training_logs:
        logger.info("分析训练性能...")
        analyzer = TrainingPerformanceAnalyzer()
        success = analyzer.analyze()
        if success:
            logger.info(f"训练性能分析完成，图表已保存到 {args.output_dir}/training 目录")
        else:
            logger.warning("训练性能分析失败，请检查训练日志文件是否存在")
    
    logger.info("性能分析完成！")
    
    # 输出图表列表
    if args.action in ['visualize', 'all']:
        print("\n生成的系统性能图表:")
        print("1. response_times_boxplot.png - 各模块响应时间分布")
        print("2. resource_usage_trend.png - 系统资源使用趋势")
        print("3. cache_hit_rates_trend.png - 各模块缓存命中率趋势")
        print("4. success_rates_trend.png - 各模块请求成功率趋势")
        print("5. throughput_trend.png - 系统吞吐量趋势")
        print("6. intent_recognition_metrics.png - 意图识别性能指标")
        print("7. entity_recognition_metrics.png - 实体识别性能指标")
        print("8. recommendation_metrics.png - 推荐系统性能指标")
        print("9. knowledge_graph_metrics.png - 知识图谱查询指标")
        print("10. performance_overview_radar.png - 系统性能总览雷达图")
    
    if (args.action in ['analyze_training', 'all'] or args.training_logs) and success:
        print("\n生成的训练性能图表:")
        print("1. ner_training_curves.png - 命名实体识别训练曲线")
        print("2. ner_convergence_speed.png - 命名实体识别收敛速度分析")
        print("3. intent_training_curves.png - 意图分类训练曲线")
        print("4. intent_overfitting_analysis.png - 意图分类过拟合分析")
        print("5. model_performance_comparison.png - 模型性能对比")
        print("6. ner_training_time.png - 命名实体识别训练时间分析")

if __name__ == "__main__":
    main() 