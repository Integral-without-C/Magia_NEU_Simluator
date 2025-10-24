import numpy as np
import matplotlib.pyplot as plt
from sympy import parse_expr, symbols
import json
from collections import defaultdict
import math



x, y, z = symbols('x y z')

def get_metric_tensor(a, b, c, alpha, beta, gamma):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    G = np.zeros((3, 3))
    G[0, 0] = a ** 2
    G[1, 1] = b ** 2
    G[2, 2] = c ** 2
    G[0, 1] = G[1, 0] = a * b * np.cos(gamma)
    G[0, 2] = G[2, 0] = a * c * np.cos(beta)
    G[1, 2] = G[2, 1] = b * c * np.cos(alpha)
    return G

def parse_symop(op_str):
    # 去掉两端空白和可能的引号、统一小写空格处理
    op_str = op_str.strip().strip("'\"")
    parts = op_str.split(',')
    if len(parts) != 3:
        raise ValueError("Invalid symmetry operation string")
    R = np.zeros((3, 3))
    t = np.zeros(3)
    for i, p in enumerate(parts):
        expr = parse_expr(p.strip())          # 直接解析表达式，保留分数形式如 1/2
        const = float(expr.subs({x: 0, y: 0, z: 0}))
        t[i] = const % 1
        for var, j in zip([x, y, z], range(3)):
            coeff = float(expr.coeff(var) or 0)
            R[i, j] = coeff
    return R, t

def pseudo_voigt(x, x0, fwhm, eta):
    """PV函数，x0为中心，fwhm为半高宽，eta为高斯/洛伦兹混合因子"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2
    return eta * (gamma**2 / ((x - x0)**2 + gamma**2)) / np.pi + \
           (1 - eta) * np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# 主程序：用户输入
print("开始读取参数...")

# 从 JSON 文件读取参数，便于批量修改
with open(r"D:\\study\\Program\\NEU_SM\\Li2MnO3.json", "r", encoding="utf-8") as f:
    params = json.load(f)

a = float(params["lattice"]["a"])
b = float(params["lattice"]["b"])
c = float(params["lattice"]["c"])
alpha = float(params["lattice"]["alpha"])
beta = float(params["lattice"]["beta"])
gamma = float(params["lattice"]["gamma"])

G = get_metric_tensor(a, b, c, alpha, beta, gamma)
G_star = np.linalg.inv(G)

# 对称操作
sym_ops_str = params.get("sym_ops", [])
sym_ops = [parse_symop(op) for op in sym_ops_str]

# 唯一旋转矩阵
point_group_R = []
for R, _ in sym_ops:
    if not any(np.allclose(R, ex, atol=1e-4) for ex in point_group_R):
        point_group_R.append(R)

# 检查是否中心对称
centro = any(np.allclose(R, -np.eye(3), atol=1e-4) for R in point_group_R)

# 不对称单元原子
asym_atoms = []
for atom in params.get("asym_atoms", []):
    asym_atoms.append({
        "label": atom["label"],
        "pos": np.array(atom["pos"], dtype=float),
        "occ": float(atom.get("occ", 1.0))
    })
print("\n不对称单元原子信息（未施加对称性）：")
print(f"{'元素':<6} {'x':>8} {'y':>8} {'z':>8} {'occ':>8}")
for atom in asym_atoms:
    label = atom['label']
    x_, y_, z_ = atom['pos']
    occ = atom['occ']
    print(f"{label:<6} {x_:>8.5f} {y_:>8.5f} {z_:>8.5f} {occ:>8.3f}")

# 中子散射长度
b_dict = {k: float(v) for k, v in params.get("scattering_lengths", {}).items()}

# Debye-Waller因子 (B_iso)
debye_waller = params.get("debye_waller", {"Li": 1.0, "Mn": 0.5, "O": 0.8})  # 默认值，如果JSON无

# 生成单元晶胞所有原子
all_atoms = []
atom_sites = defaultdict(list)  # 用于统计每个原子位点的等价位置
tol = 1e-4
for idx, atom in enumerate(asym_atoms):
    equiv_pos = []
    for R, t in sym_ops:
        new_pos = np.mod(np.dot(R, atom['pos']) + t, 1)
        if not any(np.allclose(new_pos, ex, atol=tol) for ex in equiv_pos):
            equiv_pos.append(new_pos)
    # 记录每个原子位点的所有等价位置
    atom_sites[idx] = equiv_pos
    for pos in equiv_pos:
        all_atoms.append({
            'label': atom['label'],
            'pos': pos,
            'b': b_dict[atom['label']],
            'occ': atom['occ'],
            'b_iso': debye_waller.get(atom['label'], 0.5),
            'site_idx': idx
        })

# 打印所有原胞原子信息
print("\n原胞中所有原子（对称性拓展后）：")
for idx, equiv_pos in atom_sites.items():
    atom = asym_atoms[idx]
    mult = len(equiv_pos)
    for pos in equiv_pos:
        print(f"元素: {atom['label']}, 坐标: [{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}], "
              f"B_iso: {debye_waller.get(atom['label'], 0.5)}, occ: {atom['occ']}, 位点多重度: {mult}")

# 统计原胞化学比
element_count = defaultdict(float)
for idx, equiv_pos in atom_sites.items():
    atom = asym_atoms[idx]
    mult = len(equiv_pos)
    element_count[atom['label']] += mult * atom['occ']

# 化学比化简
def gcd_list(nums):
    nums_int = [round(n * 1e6) for n in nums]  # 转为整数避免浮点误差
    g = nums_int[0]
    for n in nums_int[1:]:
        g = math.gcd(g, n)
    return g / 1e6

gcd_val = gcd_list(list(element_count.values()))
chem_ratio = {el: round(cnt / gcd_val, 3) for el, cnt in element_count.items()}

print("\n原胞化学比（最简整数比）：")
for el, cnt in chem_ratio.items():
    print(f"{el}: {cnt}")

# 波长和2theta范围
lam = float(params["experiment"].get("lambda", 1.8))
theta_min = float(params["experiment"].get("two_theta_min", 0.0))
theta_max = float(params["experiment"].get("two_theta_max", 180.0))

# 估算 hmax
sin_theta_max = np.sin(np.deg2rad(theta_max / 2))
min_d = lam / (2 * sin_theta_max) if sin_theta_max > 0 else 1e-6
hmax = int(max(a, b, c) / min_d) + 2
print(f"使用 hmax（最大晶面指数） = {hmax}")

# 计算峰
peaks = []
seen_orbits = set()
for h in range(0, hmax + 1):
    start_k = -hmax if h > 0 else 0
    for k in range(start_k, hmax + 1):
        # 把 start_l 的设置移到这里，确保 k 已定义
        start_l = -hmax if (h > 0 or k > 0) else 1  # 避免 (0,0,0)
        for l in range(start_l, hmax + 1):
            if h == 0 and k == 0 and l == 0: continue
            hkl = np.array([h, k, l])
            q2 = np.dot(hkl, np.dot(G_star, hkl))
            if q2 == 0: continue
            d = 1 / np.sqrt(q2)
            sin_theta = lam / (2 * d)
            if sin_theta > 1 or sin_theta <= 0: continue
            two_theta = 2 * np.rad2deg(np.arcsin(sin_theta))
            if two_theta < theta_min or two_theta > theta_max: continue
            # 结构因子 F
            F = 0 + 0j
            for atom in all_atoms:
                dw = np.exp(-atom['b_iso'] * (sin_theta ** 2) / (lam ** 2))  # Debye-Waller因子
                phase = np.exp(2j * np.pi * np.dot(hkl, atom['pos']))
                F += atom['b'] * atom['occ'] * phase * dw
            intens = np.abs(F) ** 2

            if intens < 1e-3: continue
            # 轨道 (等价 hkl)
            orbit = set()
            for R in point_group_R:
                hkl_p = np.dot(np.linalg.inv(R).T, hkl)
                hkl_p_rounded = np.rint(hkl_p).astype(int)
                if np.allclose(hkl_p, hkl_p_rounded, atol=1e-6):
                    orbit.add(tuple(hkl_p_rounded))
            orbit_f = frozenset(orbit)
            if orbit_f in seen_orbits: continue
            seen_orbits.add(orbit_f)
            m = len(orbit)
            # Lorentz修正
            lorentz = 1 / np.sin(np.deg2rad(two_theta))
            # Q修正
            q = 2 * np.pi / d
            I = m * intens / (q ** 2)  # 洛伦兹修正已关闭，因为与q**2修正冲突
            peaks.append((two_theta, I, (h, k, l), m))

# 排序并输出
peaks.sort(key=lambda x: x[0])
print("\n模拟峰:")
for tt, I, hkl, m in peaks:
    print(f"2θ = {tt:.2f}°, 强度 = {I:.2f}, hkl = {hkl}, 晶面多重度 = {m}")

# 用户可设置是否使用PV展宽
use_pv_broadening = params["experiment"].get("use_pv_broadening", True)  # 默认True

# 绘制
if peaks:
    two_thetas = [p[0] for p in peaks]
    intensities = [p[1] for p in peaks]
    if use_pv_broadening:
        # 构建模拟谱图
        x_grid = np.linspace(theta_min, theta_max, 2000)
        y_grid = np.zeros_like(x_grid)
        # 设置展宽参数
        fwhm = 0.1  # 半高宽，可调整
        eta = 0.5   # 混合因子，可调整
        for tt, I in zip(two_thetas, intensities):
            y_grid += I * pseudo_voigt(x_grid, tt, fwhm, eta)
        plt.plot(x_grid, y_grid, label='Pseudo-Voigt broadened')
    else:
        plt.vlines(two_thetas, 0, intensities, color='r', label='Stick pattern')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (arbitrary units)')
    plt.title('Simulated Neutron Powder Diffraction Pattern')
    plt.legend()
    plt.show()
else:
    print("无峰在范围内。")