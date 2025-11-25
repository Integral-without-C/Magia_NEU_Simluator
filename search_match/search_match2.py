import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from peaks_search2 import extract_peaks
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
import time

"""
2025.11.19
更新说明：
1. 整合PDF论文中的FoM（Figure of Merit）评分函数
2. 支持用户自定义wθ、wI、wph三个权重参数
3. 支持d值与2θ值的转换（CuKα辐射，λ=1.5406 Å）
4. 保留原有的并行计算和峰值匹配逻辑
"""

# X射线波长（CuKα辐射，单位：Å）
LAMBDA_XRAY = 1.5406

def d_to_2theta(d):
    """将d值（Å）转换为2θ值（度）"""
    if d <= 0:
        return 0.0
    theta = np.arcsin(LAMBDA_XRAY / (2 * d))
    return 2 * theta * (180 / np.pi)  # 转换为角度制

def fine_match_worker(args):
    """顶层可picklable的包装函数，用于并行计算"""
    # args: (db_path, exp_sorted, n, weights, rel_tol, abs_tol, w_theta, w_intensity, w_phase)
    return fine_match(*args)

def find_matches_peakwise(exp_peaks, db_peaks, rel_tol=0.001, abs_tol=0.005):
    """基础峰匹配逻辑（保留原实现）"""
    db_ds = [p[0] for p in db_peaks]
    db_int_map = {p[0]: p[1] for p in db_peaks}
    matched = []
    used_db_idx = set()
    
    for d_e, i_e in exp_peaks:
        tol = max(d_e * rel_tol, abs_tol)
        lo = d_e - tol
        hi = d_e + tol
        
        from bisect import bisect_left, bisect_right
        L = bisect_left(db_ds, lo)
        R = bisect_right(db_ds, hi)
        
        if L < R:
            best_score = 1e9
            best = None
            best_idx = None
            for idx in range(L, R):
                d_db = db_ds[idx]
                i_db = db_int_map[d_db]
                delta = abs(d_db - d_e)
                s = delta - 0.5 * (i_e + i_db)
                if s < best_score:
                    best_score = s
                    best = (d_db, i_db, delta)
                    best_idx = idx
            if best is not None:
                matched.append((d_e, i_e, best[0], best[1], best[2]))
                used_db_idx.add(best_idx)
    
    # 统计未匹配的强峰
    db_unmatched = [(db_ds[i], db_int_map[db_ds[i]]) for i in range(len(db_ds)) if i not in used_db_idx]
    db_unmatched_strong = sorted(db_unmatched, key=lambda x: x[1], reverse=True)[:5]
    exp_unmatched = [p for p in exp_peaks if all(abs(p[0]-m[0])>max(p[0]*rel_tol, abs_tol) for m in matched)]
    exp_unmatched_strong = sorted(exp_unmatched, key=lambda x: x[1], reverse=True)[:5]
    
    return matched, db_unmatched_strong, exp_unmatched_strong

def load_database_txt(txt_path, top_k_peaks=None):
    """加载数据库文件（保留原实现）"""
    ds, ints = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith(('最简比化学式', '空间群', '包含原子', '衍射峰位')):
                continue
            parts = ln.split()
            try:
                d = float(parts[0])
                inten = float(parts[1])
            except:
                continue
            ds.append(d)
            ints.append(inten)
    
    if len(ints) == 0:
        return [], []
    
    # 强度归一化
    ints = np.array(ints, dtype=float)
    ints = ints / (np.max(ints) if np.max(ints) > 0 else 1.0)
    peaks = sorted(zip(ds, ints), key=lambda x: x[0])
    
    # 按强度筛选Top K峰
    if top_k_peaks is not None:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:top_k_peaks]
        peaks = sorted(peaks, key=lambda x: x[0])
    
    return peaks, dict(peaks)

def plot_match(exp_d, exp_i, db_peaks, db_name, matched_pairs):
    """绘图函数（保留原实现）"""
    plt.figure(figsize=(10, 6))
    plt.plot(exp_d, exp_i, label='实验谱线', lw=2)
    db_d = [p[0] for p in db_peaks]
    db_i = [p[1] for p in db_peaks]
    plt.vlines(db_d, 0, db_i, color='orange', label='数据库谱线', lw=2)
    
    # 标记匹配峰（红=实验峰，绿=数据库峰）
    for (d_e, i_e, d_db, i_db, delta) in matched_pairs:
        plt.plot([d_e], [i_e], 'ro', markersize=6)
        plt.plot([d_db], [i_db], 'go', markersize=6)
    
    plt.title(f'匹配结果: {db_name} (FoM: {np.sqrt(np.mean([p[4] for p in matched_pairs])):.3f})' if matched_pairs else f'匹配结果: {db_name} (无匹配峰)')
    plt.xlabel('d (Angstrom) / 2θ (度)')
    plt.ylabel('归一化强度')
    plt.legend()
    plt.tight_layout()
    plt.show()

def fine_match(
    db_path, exp_sorted, n, weights, rel_tol=0.01, abs_tol=0.01,
    w_theta=0.5, w_intensity=0.5, w_phase=0.5
):
    """
    基于PDF论文FoM公式的精细匹配函数
    :param w_theta: FoM公式中2θ差异的权重（默认0.5）
    :param w_intensity: FoM公式中强度差异的权重（默认0.5）
    :param w_phase: FoM公式中实验峰匹配比例的权重（默认0.5）
    :return: (FoM分数, 数据库路径, 数据库峰列表, 匹配对列表)
    """
    # 加载数据库峰
    db_peaks, _ = load_database_txt(db_path, top_k_peaks=None)
    if not db_peaks or not exp_sorted:
        return (0.0, db_path, db_peaks, [])
    
    # 筛选实验数据范围内的数据库峰
    exp_d_min = min([d for d, _ in exp_sorted])
    exp_d_max = max([d for d, _ in exp_sorted])
    db_peaks_in_range = [p for p in db_peaks if exp_d_min <= p[0] <= exp_d_max]
    if not db_peaks_in_range:
        return (0.0, db_path, db_peaks, [])
    
    # 数据库峰按强度排序
    db_sorted = sorted(db_peaks_in_range, key=lambda x: x[1], reverse=True)
    matched_pairs = []
    used_db_idx = set()
    
    # 基础峰匹配（保留原逻辑）
    max_rank = max(len(exp_sorted), len(db_sorted)) - 1 if max(len(exp_sorted), len(db_sorted)) > 1 else 1
    for idx_exp, (d_e, i_e) in enumerate(exp_sorted[:n]):
        weight = weights[idx_exp]
        best_score = -1e9
        best_db = None
        best_delta = None
        best_db_idx = None
        
        for j, (d_db, i_db) in enumerate(db_sorted):
            if j in used_db_idx:
                continue
            delta = abs(d_e - d_db)
            tol = max(d_e * rel_tol, abs_tol)
            if delta > tol:
                continue
            
            # 基础匹配分数（用于筛选候选峰）
            delta_norm = delta / tol
            intensity_sim = min(i_e, i_db) - 0.5 * abs(i_e - i_db)
            rank_diff_norm = abs(idx_exp - j) / max_rank
            score = intensity_sim - 1.0 * delta_norm - 0.5 * rank_diff_norm
            
            if score > best_score:
                best_score = score
                best_delta = delta
                best_db = (d_db, i_db)
                best_db_idx = j
        
        if best_db is not None and best_score > 0:
            matched_pairs.append((d_e, i_e, best_db[0], best_db[1], best_delta))
            used_db_idx.add(best_db_idx)
    
    # 计算FoM公式的四个组成部分
    num_matched = len(matched_pairs)
    if num_matched == 0:
        return (0.0, db_path, db_peaks, matched_pairs)
    
    # 1. FoM_theta：2θ差异归一化值（基于d值转换）
    two_theta_exp = [d_to_2theta(d_e) for (d_e, i_e, d_db, i_db, delta) in matched_pairs]
    two_theta_db = [d_to_2theta(d_db) for (d_e, i_e, d_db, i_db, delta) in matched_pairs]
    delta_2theta = [abs(te - td) for te, td in zip(two_theta_exp, two_theta_db)]
    avg_delta_2theta = np.mean(delta_2theta)
    FoM_theta = 1.0 / (1.0 + avg_delta_2theta / 2.0)  # 归一化到[0,1]，2度为最大允许差异
    
    # 2. FoM_I：强度差异归一化值
    delta_intensity = [abs(i_e - i_db) for (d_e, i_e, d_db, i_db, delta) in matched_pairs]
    avg_delta_intensity = np.mean(delta_intensity)
    FoM_I = 1.0 - avg_delta_intensity  # 归一化到[0,1]
    
    # 3. FoM_ph：匹配实验峰强度占比
    sum_matched_exp_int = sum(i_e for (d_e, i_e, d_db, i_db, delta) in matched_pairs)
    total_exp_int = sum(i_e for (d_e, i_e) in exp_sorted)
    FoM_ph = sum_matched_exp_int / total_exp_int if total_exp_int > 0 else 0.0
    
    # 4. FoM_db：匹配数据库峰强度占比
    sum_matched_db_int = sum(i_db for (d_e, i_e, d_db, i_db, delta) in matched_pairs)
    total_db_int = sum(i_db for (d_db, i_db) in db_peaks_in_range)
    FoM_db = sum_matched_db_int / total_db_int if total_db_int > 0 else 0.0
    
    # 计算FoM最终分数（严格遵循PDF公式）
    total_weight = w_theta + w_intensity + w_phase
    if total_weight == 0:
        weighted_sum = 0.0
    else:
        weighted_sum = (w_theta * FoM_theta + w_intensity * FoM_I + w_phase * FoM_ph) / total_weight
    
    FoM_score = np.sqrt(FoM_db * weighted_sum) if (FoM_db * weighted_sum) >= 0 else 0.0
    
    # 保留原有的三强峰未匹配惩罚（可选，可调整惩罚系数）
    db_top3 = db_sorted[:3]
    unmatched_penalty = 0.0
    for idx, (d_db, i_db) in enumerate(db_top3):
        matched_flag = any(abs(d_db - mp[2]) < max(d_db * rel_tol, abs_tol) for mp in matched_pairs)
        if not matched_flag:
            penalty_weight = weights[idx] if idx < len(weights) else 0.3
            unmatched_penalty += penalty_weight * i_db
    
    # 应用惩罚项（惩罚系数2可调整）
    # FoM_score = max(0.0, FoM_score - unmatched_penalty * 0.5) 
    FoM_score =  FoM_score - unmatched_penalty * 2   # 可允许负分数以区分差异较大情况
    
    return (FoM_score, db_path, db_peaks, matched_pairs)

def main(
    exp_file, db_folder, top_n=100, rel_tol=0.05, abs_tol=0.05,
    use_background=False, w_theta=0.5, w_intensity=0.5, w_phase=0.5
):
    """
    主函数：整合实验数据处理、数据库匹配、结果输出
    :param w_theta: FoM公式中2θ差异的权重（用户可修改）
    :param w_intensity: FoM公式中强度差异的权重（用户可修改）
    :param w_phase: FoM公式中实验峰匹配比例的权重（用户可修改）
    """
    # 1. 实验数据寻峰
    peaks, d, intensity, *_ = extract_peaks(
        exp_file, top_n=100, threshold=0.2, use_background=use_background
    )
    exp_peaks = sorted(peaks, key=lambda x: x[0])
    exp_d = d
    exp_i = intensity / (np.max(intensity) if np.max(intensity) > 0 else 1.0)
    exp_sorted = sorted(exp_peaks, key=lambda x: x[1], reverse=True)
    
    # 2. 配置匹配参数
    n = min(10, len(exp_sorted))  # 取前10个最强实验峰进行匹配
    weights = np.linspace(1.0, 0.3, n)  # 峰强度排名权重（与FoM权重无关）
    
    # 3. 获取数据库文件列表
    db_files = [os.path.join(db_folder, f) for f in os.listdir(db_folder) if f.endswith('.txt')]
    if not db_files:
        print("数据库文件夹中未找到.txt格式的数据库文件！")
        return
    
    # 4. 并行执行精细匹配（传递FoM权重参数）
    args_list = [
        (p, exp_sorted, n, weights, rel_tol, abs_tol, w_theta, w_intensity, w_phase)
        for p in db_files
    ]
    
    max_workers = min(os.cpu_count() or 6, 10)  # 自适应CPU核心数

    # 并行执行精细匹配：使用 futures + as_completed 来显示文本进度条和已用/预计时间
    args_list = [
        (p, exp_sorted, n, weights, rel_tol, abs_tol, w_theta, w_intensity, w_phase)
        for p in db_files
    ]

    start_time = time.time()
    fine_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fine_match_worker, args) for args in args_list]
        total = len(futures)
        if total == 0:
            print("无可用数据库文件进行匹配。")
            return

        # 文本进度条参数
        bar_len = 40

        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                res = fut.result()
            except Exception as e:
                res = None
            fine_results.append(res)

            elapsed = time.time() - start_time
            percent = completed / total
            filled = int(percent * bar_len)
            bar = '#' * filled + '-' * (bar_len - filled)
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0

            def _fmt(t):
                return time.strftime("%H:%M:%S", time.gmtime(t))

            print(f"\r搜索匹配进度: [{bar}] {completed}/{total} ({percent*100:5.1f}%) 已用: {_fmt(elapsed)} 预计剩余: {_fmt(eta)}", end='', flush=True)

        # 换行以结束进度行
        print()
    
    # 5. 排序并输出结果
    fine_results = [r for r in fine_results if r is not None]
    fine_results.sort(reverse=True, key=lambda x: x[0])  # 按FoM分数降序排列
    
    print(f"\n匹配结果（Top {min(top_n, len(fine_results))}）：")
    print("-" * 80)
    print(f"{'排名':<5} {'数据库文件':<40} {'FoM分数':<20} {'匹配峰数':<10}")
    print("-" * 80)
    
    for i in range(min(top_n, len(fine_results))):
        score, db_path, db_peaks, matched = fine_results[i]
        db_name = os.path.basename(db_path)
        print(f"{i+1:<5} {db_name:<47} {score:.5f} {'':<20} {len(matched):<10}")
        
        # 绘制前5个最佳匹配的谱图
        if i < 5:
            plot_match(exp_d, exp_i, db_peaks, db_name, matched)

if __name__ == '__main__':
    # 用户可修改以下参数（重点：w_theta、w_intensity、w_phase）
    main(
        exp_file=r"D:\study\Program\NEU_SM\TOF_convert_to_d\normalized_d_intensity.dat",
        db_folder=r"D:\study\Program\NEU_SM\TOF_convert_to_d\SMtest\txt",
        top_n=20,  # 输出Top 20匹配结果
        use_background=True,  # 是否使用背景扣除
        w_theta=0.9,  # FoM中2θ差异权重（默认0.5）
        w_intensity=0.6,  # FoM中强度差异权重（默认0.5）
        w_phase=0.5  # FoM中实验峰匹配比例权重（默认0.5）
    )