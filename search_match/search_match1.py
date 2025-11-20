import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from peaks_search2 import extract_peaks
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False
'''
2025.11.19
更新人：专业混分
'''


def fine_match_worker(args):
    """
    顶层可 picklable 的包装函数，用于并行。
    args: (db_path, exp_sorted, n, weights, rel_tol, abs_tol)
    """
    return fine_match(*args)


def find_matches_peakwise(exp_peaks, db_peaks, rel_tol=0.001, abs_tol=0.005):
    """
    exp_peaks/db_peaks: list of (d, inten) sorted by d。
    返回 matched_pairs list: [(d_exp, inten_exp, d_db, inten_db, delta_d), ...]
    """
    db_ds = [p[0] for p in db_peaks]
    db_int_map = {p[0]:p[1] for p in db_peaks}
    matched = []
    used_db_idx = set()
    for d_e, i_e in exp_peaks:
        tol = max(d_e * rel_tol, abs_tol)
        lo = d_e - tol; hi = d_e + tol
        # 二分查找
        from bisect import bisect_left, bisect_right
        L = bisect_left(db_ds, lo); R = bisect_right(db_ds, hi)
        if L < R:
            # 选最近的 db 峰
            cand_idxs = list(range(L, R))
            best = None; best_score = 1e9; best_idx = None
            for idx in cand_idxs:
                d_db = db_ds[idx]; i_db = db_int_map[d_db]
                delta = abs(d_db - d_e)
                s = delta - 0.5*(i_e + i_db)
                if s < best_score:
                    best_score = s; best = (d_db, i_db, delta); best_idx = idx
            if best is not None:
                matched.append((d_e, i_e, best[0], best[1], best[2]))
                used_db_idx.add(best_idx)
    # 可选：返回未匹配强峰
    db_unmatched = [(db_ds[i], db_int_map[db_ds[i]]) for i in range(len(db_ds)) if i not in used_db_idx]
    db_unmatched_strong = sorted(db_unmatched, key=lambda x: x[1], reverse=True)[:5]
    exp_unmatched = [p for p in exp_peaks if all(abs(p[0]-m[0])>max(p[0]*rel_tol, abs_tol) for m in matched)]
    exp_unmatched_strong = sorted(exp_unmatched, key=lambda x: x[1], reverse=True)[:5]
    return matched, db_unmatched_strong, exp_unmatched_strong

def load_database_txt(txt_path, top_k_peaks=None):
    ds, ints = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('最简比化学式') or ln.startswith('空间群') or ln.startswith('包含原子') or ln.startswith('衍射峰位'):
                continue
            parts = ln.split()
            try:
                d = float(parts[0]); inten = float(parts[1])
            except:
                continue
            ds.append(d); ints.append(inten)
    if len(ints)==0:
        return [], []
    ints = np.array(ints, dtype=float)
    # 归一化
    ints = ints / (np.max(ints) if np.max(ints)>0 else 1.0)
    peaks = sorted(zip(ds, ints), key=lambda x: x[0])
    if top_k_peaks is not None:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:top_k_peaks]
        peaks = sorted(peaks, key=lambda x: x[0])
    return peaks, dict(peaks)

def plot_match(exp_d, exp_i, db_peaks, db_name, matched_pairs):
    plt.figure(figsize=(10,6))
    plt.plot(exp_d, exp_i, label='实验谱线', lw=2)
    db_d = [p[0] for p in db_peaks]
    db_i = [p[1] for p in db_peaks]
    plt.vlines(db_d, 0, db_i, color='orange', label='数据库谱线', lw=2)
    # 标记匹配峰
    for (d_e, i_e, d_db, i_db, delta) in matched_pairs:
        plt.plot([d_e], [i_e], 'ro')
        plt.plot([d_db], [i_db], 'go')
    plt.title(f'匹配结果: {db_name}')
    plt.xlabel('d (Angstrom)')
    plt.ylabel('归一化强度')
    plt.legend()
    plt.tight_layout()
    plt.show()

def coarse_score(db_path, exp_top, rel_tol=0.01, abs_tol=0.01):
    db_peaks, _ = load_database_txt(db_path, top_k_peaks=None)
    if not db_peaks: return (0.0, db_path)
    # 只考虑实验范围内的数据库峰
    exp_d_min = min([d for d, _ in exp_top])
    exp_d_max = max([d for d, _ in exp_top])
    db_peaks_in_range = [p for p in db_peaks if exp_d_min <= p[0] <= exp_d_max]
    if not db_peaks_in_range:
        return (0.0, db_path)
    # 用匹配峰数和强度作为分数
    matched, _, _ = find_matches_peakwise(exp_top, db_peaks_in_range, rel_tol, abs_tol)
    score = len(matched) + sum([min(i_e, i_db) for (_, i_e, _, i_db, _) in matched])
    return (score, db_path)

def fine_match(db_path, exp_sorted, n, weights, rel_tol=0.01, abs_tol=0.01):
    db_peaks, _ = load_database_txt(db_path, top_k_peaks=None)
    if not db_peaks or not exp_sorted:
        return (0, db_path, db_peaks, [])
    # 1. 只保留实验数据范围内的数据库峰
    exp_d_min = min([d for d, _ in exp_sorted])
    exp_d_max = max([d for d, _ in exp_sorted])
    db_peaks_in_range = [p for p in db_peaks if exp_d_min <= p[0] <= exp_d_max]
    if not db_peaks_in_range:
        return (0, db_path, db_peaks, [])
    db_sorted = sorted(db_peaks_in_range, key=lambda x: x[1], reverse=True)
    matched_pairs = []
    used_db_idx = set()
    total_score = 0.0
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
            # Normalized penalties
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
            total_score += weight * best_score
    # Bonus for number of matches
    total_score += len(matched_pairs) * 0.1

    # 2. 惩罚数据库三强峰未匹配
    db_top3 = db_sorted[:3]
    unmatched_penalty = 0.0
    for idx, (d_db, i_db) in enumerate(db_top3):
        # 判断是否被匹配
        matched = any(abs(d_db - mp[2]) < max(d_db * rel_tol, abs_tol) for mp in matched_pairs)
        if not matched:
            # 权重与强度相关
            penalty_weight = weights[idx] if idx < len(weights) else 0.3
            unmatched_penalty += penalty_weight * i_db
    total_score -= unmatched_penalty

    return (total_score, db_path, db_peaks, matched_pairs)

def main(exp_file, db_folder, top_n=100, rel_tol=0.05, abs_tol=0.05, use_background=False):
    # 1. 实验数据寻峰
    peaks, d, intensity, *_ = extract_peaks(
        exp_file, top_n=100, threshold=0.2, use_background=use_background
    )
    exp_peaks = sorted(peaks, key=lambda x: x[0])
    exp_d = d
    exp_i = intensity / (np.max(intensity) if np.max(intensity)>0 else 1.0)
    exp_sorted = sorted(exp_peaks, key=lambda x: x[1], reverse=True)
    exp_top3 = exp_sorted[:3]
    n = min(10, len(exp_sorted))
    weights = np.linspace(1.0, 0.3, n)

    # 2. 获取数据库文件列表（不再粗筛，直接精筛所有文件）
    db_files = [os.path.join(db_folder, f) for f in os.listdir(db_folder) if f.endswith('.txt')]

    # 构造参数列表，避免使用 lambda（lambda 在子进程中不可 pickle）
    args_list = [(p, exp_sorted, n, weights, rel_tol, abs_tol) for p in db_files]

    # 直接对所有文件并行执行 fine_match（使用进程池）
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    max_workers = min(os.cpu_count() or 6, 10)  # 根据机器调整上限，建议此值
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        fine_results = list(pool.map(fine_match_worker, args_list))

    # 排序并输出 top_n
    fine_results = [r for r in fine_results if r is not None]
    fine_results.sort(reverse=True)
    for i in range(min(top_n, len(fine_results))):
        score, db_path, db_peaks, matched = fine_results[i]
        print(f"Top {i+1}: {os.path.basename(db_path)}, 匹配分数: {score}, 匹配峰数: {len(matched)}")
        plot_match(exp_d, exp_i, db_peaks, os.path.basename(db_path), matched)
# ...existing code...

if __name__ == '__main__':
    main(
        exp_file=r"D:\study\Program\NEU_SM\TOF_convert_to_d\normalized_d_intensityRUN0000260_groupBS.dat",
        db_folder=r"D:\study\Program\NEU_SM\TOF_convert_to_d\SMtest\txt",
        top_n=20,
        use_background=False  # 用户可改为 False 或 True
    )