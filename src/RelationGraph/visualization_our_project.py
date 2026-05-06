from func.utils.calc_matrix import load_affinity_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_affinity_heatmap(affinity_matrix_path, title="亲缘强度热度图"):

    affinity_matrix, class_names = load_affinity_matrix(affinity_matrix_path)

    mat = affinity_matrix.copy().astype(float)  # 将数据类型转换为 float
    n = len(class_names)  # 得到分类数
    np.fill_diagonal(mat, np.nan)  # 把对角线置为 NaN（对角线不具有实际意义）

    non_diag = mat[~np.eye(n, dtype=bool)]  # 得到非对角线元素为 True 的布尔掩码
    min_val, max_val = np.nanmin(non_diag), np.nanmax(non_diag)  # 计算最大值和最小值
    if max_val > min_val:
        mat = (mat - min_val) / (max_val - min_val)  # 归一化，得到的是0~1之间的值
    else:
        mat = mat - min_val  # 常量矩阵归零

    annot_mat = np.empty_like(mat, dtype=object)  # 创建一个与 mat 形状相同、类型为 object 的空数组，存储热力图上每个单元格的注释文本
    for i in range(n):  # 向 annot_mat 写入值
        for j in range(n):
            if np.isnan(mat[i, j]):
                annot_mat[i, j] = ''  # 如果是 NaN，赋空值
            else:
                annot_mat[i, j] = f'{mat[i, j]:.4f}'  # 否则保留2位小数

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                                       'Arial Unicode MS']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    # --- 绘图 ---#
    fig, ax = plt.subplots(1, 1, figsize=(n * 0.7, n * 0.7))

    ax = sns.heatmap(mat, annot=annot_mat, fmt='', cmap='YlOrRd',  # 颜色映射表
                     vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names,  # vmin，vmax 颜色映射的数值范围
                     annot_kws={'size': 8}, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})  # linewidths： 单元格之间的分割线宽度为 0.5 磅
    # 绘制热力图

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('类别', fontsize=10)
    plt.ylabel('类别', fontsize=10)
    # 设置轴标签与标题

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    # 设置刻度标签，x轴有45的旋转

    cbar = ax.collections[0].colorbar
    cbar.set_label('亲缘强度', fontsize=12, labelpad=10)
    # 给颜色条标签也设置为中文字体

    plt.tight_layout()
    plt.savefig('affinity_heatmap.png', dpi=300, bbox_inches='tight') # 保存图片
    plt.show()# 打印出图片


def print_top_affinity_pairs(affinity_matrix_path, top_k = 10):
    # Step 1：加载矩阵和类别名
    affinity_matrix, class_names = load_affinity_matrix(affinity_matrix_path)
    n = len(class_names)

    # Step 2：提取上三角的非对角线元素（避免 (i,j) 和 (j,i) 重复）
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):  # 只取 i < j 的部分
            val = affinity_matrix[i, j]
            pairs.append((val, i, j))

    # Step 3：按亲缘强度降序排序，并截取前top_k个
    pairs.sort(key=lambda x: x[0], reverse=True)
    top10 = pairs[:top_k]

    # Step 4：打印结果
    print(f"亲缘强度 Top {len(top10)} 岗位对 (来自 {affinity_matrix_path})：")
    print("-" * 30)
    for rank, (val, i, j) in enumerate(top10, 1):
        name_i = class_names[i]
        name_j = class_names[j]
        print(f"{rank:2d}. {name_i}  <-->  {name_j}  亲缘强度: {val:.4f}")

plot_affinity_heatmap("affinity_matrix_lora.json")
print_top_affinity_pairs("affinity_matrix_lora.json", 10)