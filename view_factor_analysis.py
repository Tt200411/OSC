#!/usr/bin/env python3
"""
快速查看ETT数据集每日因子分析结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 路径配置
analysis_path = Path(__file__).parent / "Daily_Factors_Analysis"

def print_header(text, char='='):
    """打印标题"""
    width = 80
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def show_dataset_summary(dataset_name):
    """显示单个数据集的摘要"""
    csv_file = analysis_path / f"{dataset_name}_daily_factors.csv"

    if not csv_file.exists():
        print(f"错误: 找不到文件 {csv_file}")
        return

    df = pd.read_csv(csv_file)

    print_header(f"{dataset_name} 每日因子分析", '-')

    print(f"数据天数: {len(df)} 天")
    print(f"日期范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")

    # 关键因子统计
    key_factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

    print("\n关键因子统计:")
    print("-" * 80)
    print(f"{'因子':<15} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print("-" * 80)

    for factor in key_factors:
        mean_val = df[factor].mean()
        std_val = df[factor].std()
        min_val = df[factor].min()
        max_val = df[factor].max()
        print(f"{factor:<15} {mean_val:>12.4f} {std_val:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")

    # 极值日期
    print("\n极值日期:")
    print("-" * 80)

    # 最高均值日
    max_mean_idx = df['mean'].idxmax()
    print(f"最高平均负荷日: {df.loc[max_mean_idx, 'date']} (mean={df.loc[max_mean_idx, 'mean']:.2f})")

    # 最低均值日
    min_mean_idx = df['mean'].idxmin()
    print(f"最低平均负荷日: {df.loc[min_mean_idx, 'date']} (mean={df.loc[min_mean_idx, 'mean']:.2f})")

    # 最高波动日
    max_vol_idx = df['volatility'].idxmax()
    print(f"最高波动日: {df.loc[max_vol_idx, 'date']} (volatility={df.loc[max_vol_idx, 'volatility']:.2f})")

    # 最稳定日
    min_std_idx = df['std'].idxmin()
    print(f"最稳定日: {df.loc[min_std_idx, 'date']} (std={df.loc[min_std_idx, 'std']:.2f})")

    # 趋势分析
    print("\n趋势分析:")
    print("-" * 80)
    uptrend_days = len(df[df['trend'] > 0])
    downtrend_days = len(df[df['trend'] < 0])
    flat_days = len(df[df['trend'] == 0])

    print(f"上升趋势日: {uptrend_days} 天 ({uptrend_days/len(df)*100:.1f}%)")
    print(f"下降趋势日: {downtrend_days} 天 ({downtrend_days/len(df)*100:.1f}%)")
    print(f"平稳日: {flat_days} 天 ({flat_days/len(df)*100:.1f}%)")

    # 波动性分类
    print("\n波动性分类 (基于volatility):")
    print("-" * 80)
    q25 = df['volatility'].quantile(0.25)
    q75 = df['volatility'].quantile(0.75)

    low_vol = len(df[df['volatility'] <= q25])
    med_vol = len(df[(df['volatility'] > q25) & (df['volatility'] <= q75)])
    high_vol = len(df[df['volatility'] > q75])

    print(f"低波动日 (≤Q1={q25:.2f}): {low_vol} 天 ({low_vol/len(df)*100:.1f}%)")
    print(f"中波动日 (Q1-Q3): {med_vol} 天 ({med_vol/len(df)*100:.1f}%)")
    print(f"高波动日 (>Q3={q75:.2f}): {high_vol} 天 ({high_vol/len(df)*100:.1f}%)")

def show_comparison():
    """显示所有数据集对比"""
    summary_file = analysis_path / "all_datasets_summary.csv"

    if not summary_file.exists():
        print(f"错误: 找不到汇总文件 {summary_file}")
        return

    df = pd.read_csv(summary_file)

    print_header("所有数据集对比分析")

    # 按因子分组展示
    factors = df['Factor'].unique()

    for factor in factors:
        print(f"\n【{factor.upper()}】")
        print("-" * 80)
        factor_data = df[df['Factor'] == factor].sort_values('Mean')

        for _, row in factor_data.iterrows():
            dataset = row['Dataset']
            mean_val = row['Mean']
            std_val = row['Std']
            min_val = row['Min']
            max_val = row['Max']

            print(f"{dataset:8s}: 均值={mean_val:10.4f} (±{std_val:8.4f})  "
                  f"范围=[{min_val:8.4f}, {max_val:8.4f}]")

def show_top_days(dataset_name, factor, n=10, ascending=False):
    """显示指定因子的Top N天"""
    csv_file = analysis_path / f"{dataset_name}_daily_factors.csv"

    if not csv_file.exists():
        print(f"错误: 找不到文件 {csv_file}")
        return

    df = pd.read_csv(csv_file)

    if factor not in df.columns:
        print(f"错误: 因子 '{factor}' 不存在")
        print(f"可用因子: {', '.join(df.columns)}")
        return

    order = "最低" if ascending else "最高"
    print_header(f"{dataset_name} - {factor.upper()} {order} {n} 天", '-')

    top_df = df.nlargest(n, factor) if not ascending else df.nsmallest(n, factor)

    print(f"{'排名':<6} {'日期':<20} {factor.upper():<12} {'均值':<12} {'标准差':<12}")
    print("-" * 80)

    for idx, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"{idx:<6} {row['date']:<20} {row[factor]:<12.4f} {row['mean']:<12.4f} {row['std']:<12.4f}")

def show_correlation_analysis(dataset_name):
    """显示因子相关性分析"""
    csv_file = analysis_path / f"{dataset_name}_daily_factors.csv"

    if not csv_file.exists():
        print(f"错误: 找不到文件 {csv_file}")
        return

    df = pd.read_csv(csv_file)

    print_header(f"{dataset_name} 因子相关性分析", '-')

    # 选择数值型因子
    numeric_factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range',
                      'skewness', 'kurtosis', 'iqr']

    corr_matrix = df[numeric_factors].corr()

    # 找出高相关性对（绝对值>0.7）
    print("高相关性因子对 (|r| > 0.7):")
    print("-" * 80)

    high_corr = []
    for i in range(len(numeric_factors)):
        for j in range(i+1, len(numeric_factors)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append((numeric_factors[i], numeric_factors[j], corr_val))

    if high_corr:
        for f1, f2, corr_val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"{f1:12s} <-> {f2:12s}: r = {corr_val:7.4f}")
    else:
        print("未发现高相关性因子对")

def show_seasonal_analysis(dataset_name):
    """显示季节性分析"""
    csv_file = analysis_path / f"{dataset_name}_daily_factors.csv"

    if not csv_file.exists():
        print(f"错误: 找不到文件 {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({
        12: '冬', 1: '冬', 2: '冬',
        3: '春', 4: '春', 5: '春',
        6: '夏', 7: '夏', 8: '夏',
        9: '秋', 10: '秋', 11: '秋'
    })

    print_header(f"{dataset_name} 季节性分析", '-')

    # 按季节统计
    seasonal_stats = df.groupby('season')[['mean', 'std', 'volatility']].agg(['mean', 'std'])

    print("季节性统计:")
    print("-" * 80)

    for season in ['春', '夏', '秋', '冬']:
        if season in seasonal_stats.index:
            print(f"\n{season}季:")
            print(f"  平均负荷: {seasonal_stats.loc[season, ('mean', 'mean')]:.2f} (±{seasonal_stats.loc[season, ('mean', 'std')]:.2f})")
            print(f"  平均标准差: {seasonal_stats.loc[season, ('std', 'mean')]:.2f} (±{seasonal_stats.loc[season, ('std', 'std')]:.2f})")
            print(f"  平均波动率: {seasonal_stats.loc[season, ('volatility', 'mean')]:.2f} (±{seasonal_stats.loc[season, ('volatility', 'std')]:.2f})")

def main():
    """主函数"""
    if not analysis_path.exists():
        print(f"错误: 分析目录不存在 {analysis_path}")
        print("请先运行 calculate_daily_factors.py 生成因子数据")
        return

    print_header("ETT数据集每日因子分析查看器")

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'summary':
            # 显示所有数据集摘要
            for dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                show_dataset_summary(dataset)

        elif command == 'compare':
            # 显示对比分析
            show_comparison()

        elif command == 'top':
            # 显示Top N天
            if len(sys.argv) < 4:
                print("用法: python view_factor_analysis.py top <dataset> <factor> [n] [asc/desc]")
                print("示例: python view_factor_analysis.py top ETTh1 volatility 10 desc")
                return

            dataset = sys.argv[2]
            factor = sys.argv[3]
            n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
            ascending = sys.argv[5].lower() == 'asc' if len(sys.argv) > 5 else False

            show_top_days(dataset, factor, n, ascending)

        elif command == 'corr':
            # 显示相关性分析
            if len(sys.argv) < 3:
                print("用法: python view_factor_analysis.py corr <dataset>")
                return

            dataset = sys.argv[2]
            show_correlation_analysis(dataset)

        elif command == 'season':
            # 显示季节性分析
            if len(sys.argv) < 3:
                print("用法: python view_factor_analysis.py season <dataset>")
                return

            dataset = sys.argv[2]
            show_seasonal_analysis(dataset)

        else:
            print(f"未知命令: {command}")
            print_usage()

    else:
        # 默认显示所有摘要
        for dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            show_dataset_summary(dataset)

        print("\n")
        show_comparison()

def print_usage():
    """打印使用说明"""
    print("\n使用说明:")
    print("-" * 80)
    print("python view_factor_analysis.py                    # 显示所有数据集摘要")
    print("python view_factor_analysis.py summary            # 显示所有数据集详细摘要")
    print("python view_factor_analysis.py compare            # 显示数据集对比")
    print("python view_factor_analysis.py top <dataset> <factor> [n] [asc/desc]")
    print("                                                  # 显示Top N天")
    print("python view_factor_analysis.py corr <dataset>     # 显示相关性分析")
    print("python view_factor_analysis.py season <dataset>   # 显示季节性分析")
    print("\n示例:")
    print("  python view_factor_analysis.py top ETTh1 volatility 10 desc")
    print("  python view_factor_analysis.py corr ETTm1")
    print("  python view_factor_analysis.py season ETTh2")

if __name__ == "__main__":
    main()
