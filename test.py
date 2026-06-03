import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto  # scipy的广义帕累托分布工具

# -------------------------- 1. 设置核心参数 --------------------------
# 固定位置参数μ=0、尺度参数σ=1，仅改变形状参数ξ（核心差异）
mu = 0    # 位置参数：分布起点
sigma = 1 # 尺度参数：控制分布离散程度
xi_values = [3, 0, -0.3]  # 三种典型形状参数：厚尾、指数尾、薄尾
colors = ['#2E86AB', '#A23B72', '#F18F01']  # 配色（区分三种分布）
labels = [r'$\xi=0.5$ (厚尾)', r'$\xi=0$ (指数尾)', r'$\xi=-0.2$ (薄尾)']

# -------------------------- 2. 生成绘图数据 --------------------------
# 定义x轴范围（适配不同ξ的支撑域）
x_thick = np.linspace(mu, 10, 1000)          # ξ=0.5（厚尾）：无上限，取0-10
x_exp = np.linspace(mu, 10, 1000)            # ξ=0（指数尾）：无上限，取0-10
x_thin = np.linspace(mu, mu - sigma/xi_values[2], 1000)  # ξ=-0.2（薄尾）：有上限，计算上限值

# 计算每种ξ对应的概率密度函数（PDF）值
pdf_thick = genpareto.pdf(x_thick, xi_values[0], loc=mu, scale=sigma)
pdf_exp = genpareto.pdf(x_exp, xi_values[1], loc=mu, scale=sigma)
pdf_thin = genpareto.pdf(x_thin, xi_values[2], loc=mu, scale=sigma)

# -------------------------- 3. 绘图展示 --------------------------
plt.figure(figsize=(10, 6), dpi=100)

# 绘制三条PDF曲线
plt.plot(x_thick, pdf_thick, color=colors[0], label=labels[0], linewidth=2.5)
plt.plot(x_exp, pdf_exp, color=colors[1], label=labels[1], linewidth=2.5)
plt.plot(x_thin, pdf_thin, color=colors[2], label=labels[2], linewidth=2.5)

# 图表美化与标注
plt.title('广义帕累托分布（GPD）不同形状参数的形态对比', fontsize=14, pad=15)
plt.xlabel('x (变量值)', fontsize=12)
plt.ylabel('PDF (概率密度)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(-0.5, 10.5)  # 统一x轴范围，便于对比
plt.ylim(0, 1.1)      # 统一y轴范围

# 标注薄尾分布的上限
plt.axvline(x=mu - sigma/xi_values[2], color=colors[2], linestyle=':', alpha=0.7)
plt.text(mu - sigma/xi_values[2] + 0.2, 0.1, f'上限={mu - sigma/xi_values[2]:.1f}', color=colors[2])

plt.tight_layout()
plt.show()