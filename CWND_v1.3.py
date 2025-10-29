#CWND_v1.3.py
## 1.增加批处理功能，可以一次处理多个 CIF 文件，并生成对应的模拟图和数据文件。
## 2.改进了文件命名，清理非法字符，避免保存文件时出错。
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 批处理保存图片，不弹出窗口
import matplotlib.pyplot as plt
from sympy import parse_expr, symbols
from collections import defaultdict
import math
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# 从 CIF_reading 导入
from CIF_reading import (
    parse_cif_text,
    get_cell_params,
    get_symmetry_operations,
    get_atom_sites,
    get_space_group,
    list_cif_files
)

# Sears: International Tables for Crystallography Vol. C, Sec. 4.4.4
# 天然元素的束缚相干中子散射长度（实部, 虚部，单位：fm）
NATURAL_SCATTERING_LENGTHS = {
    "H": (-3.739, 0.0), "He": (3.26, 0.0), "Li": (-1.9, 0.0), "Be": (7.79, 0.0),
    "B": (5.3, 0.213), "C": (6.646, 0.0), "N": (9.36, 0.0), "O": (5.803, 0.0),
    "F": (5.654, 0.0), "Ne": (4.566, 0.0), "Na": (3.63, 0.0), "Mg": (5.375, 0.0),
    "Al": (3.449, 0.0), "Si": (4.1491, 0.0), "P": (5.13, 0.0), "S": (2.847, 0.0),
    "Cl": (9.577, 0.0), "Ar": (1.909, 0.0), "K": (3.67, 0.0), "Ca": (4.7, 0.0),
    "Sc": (12.29, 0.0), "Ti": (-3.37, 0.0), "V": (-0.3824, 0.0), "Cr": (3.635, 0.0),
    "Mn": (-3.75, 0.0), "Fe": (9.45, 0.0), "Co": (2.49, 0.0), "Ni": (10.3, 0.0),
    "Cu": (7.718, 0.0), "Zn": (5.6, 0.0), "Ga": (7.288, 0.0), "Ge": (8.185, 0.0),
    "As": (6.58, 0.0), "Se": (7.97, 0.0), "Br": (6.795, 0.0), "Kr": (7.81, 0.0),
    "Rb": (7.09, 0.0), "Sr": (7.02, 0.0), "Y": (7.75, 0.0), "Zr": (7.16, 0.0),
    "Nb": (7.054, 0.0), "Mo": (6.715, 0.0), "Tc": (6.8, 0.0), "Ru": (7.03, 0.0),
    "Rh": (5.88, 0.0), "Pd": (5.91, 0.0), "Ag": (5.922, 0.0),
    "Cd": (4.87, -0.7), "In": (2.08, -0.0539), "Sn": (6.225, 0.0), "Sb": (5.57, 0.0),
    "Te": (5.8, 0.0), "I": (5.28, 0.0), "Xe": (4.92, 0.0), "Cs": (5.42, 0.0),
    "Ba": (5.07, 0.0), "La": (8.24, 0.0), "Ce": (4.84, 0.0), "Pr": (4.58, 0.0),
    "Nd": (7.69, 0.0), "Pm": (12.6, 0.0),
    "Sm": (0.8, -1.65), "Eu": (7.22, -1.26), "Gd": (6.5, -13.82),
    "Tb": (7.38, 0.0), "Dy": (16.9, -0.276), "Ho": (8.01, 0.0), "Er": (7.79, 0.0),
    "Tm": (7.07, 0.0), "Yb": (12.43, 0.0), "Lu": (7.21, 0.0), "Hf": (7.77, 0.0),
    "Ta": (6.91, 0.0), "W": (4.86, 0.0), "Re": (9.2, 0.0), "Os": (10.7, 0.0),
    "Ir": (10.6, 0.0), "Pt": (9.6, 0.0), "Au": (7.63, 0.0), "Hg": (12.692, 0.0),
    "Tl": (8.776, 0.0), "Pb": (9.405, 0.0), "Bi": (8.532, 0.0), "Ra": (10.0, 0.0),
    "Th": (10.31, 0.0), "Pa": (9.1, 0.0), "U": (8.417, 0.0), "Np": (10.55, 0.0),
    "Pu": (14.1, 0.0), "Am": (8.3, 0.0), "Cm": (9.5, 0.0)
}

# 同位素的束缚相干中子散射长度（real, imag），单位：fm
ISOTOPE_SCATTERING_LENGTHS = {
    "1-H": (-3.7406, 0.0), "2-H": (6.671, 0.0), "3-H": (4.792, 0.0),
    "3-He": (5.74, -1.483), "4-He": (3.26, 0.0),
    "6-Li": (2.0, -0.261), "7-Li": (-2.22, 0.0),
    "9-Be": (7.79, 0.0),
    "10-B": (-0.1, 1.066), "11-B": (6.65, 0.0),
    "12-C": (6.6511, 0.0), "13-C": (6.19, 0.0),
    "14-N": (9.37, 0.0), "15-N": (6.44, 0.0),
    "16-O": (5.803, 0.0), "17-O": (5.78, 0.0), "18-O": (5.84, 0.0),
    "19-F": (5.654, 0.0),
    "20-Ne": (4.631, 0.0), "21-Ne": (6.66, 0.0), "22-Ne": (3.87, 0.0),
    "23-Na": (3.63, 0.0),
    "24-Mg": (5.66, 0.0), "25-Mg": (3.62, 0.0), "26-Mg": (4.89, 0.0),
    "27-Al": (3.449, 0.0),
    "28-Si": (4.107, 0.0), "29-Si": (4.7, 0.0), "30-Si": (4.58, 0.0),
    "31-P": (5.13, 0.0),
    "32-S": (2.804, 0.0), "33-S": (4.74, 0.0), "34-S": (3.48, 0.0), "36-S": (3.0, 0.0),
    "35-Cl": (11.65, 0.0), "37-Cl": (3.08, 0.0),
    "36-Ar": (24.9, 0.0), "38-Ar": (3.5, 0.0), "40-Ar": (1.83, 0.0),
    "39-K": (3.74, 0.0), "40-K": (3.0, 0.0), "41-K": (2.69, 0.0),
    "40-Ca": (4.8, 0.0), "42-Ca": (3.36, 0.0), "43-Ca": (-1.56, 0.0), "44-Ca": (1.42, 0.0), "46-Ca": (3.6, 0.0), "48-Ca": (0.39, 0.0),
    "45-Sc": (12.29, 0.0),
    "46-Ti": (4.725, 0.0), "47-Ti": (3.53, 0.0), "48-Ti": (-5.86, 0.0), "49-Ti": (0.98, 0.0), "50-Ti": (5.88, 0.0),
    "50-V": (7.6, 0.0), "51-V": (-0.402, 0.0),
    "50-Cr": (-4.5, 0.0), "52-Cr": (4.92, 0.0), "53-Cr": (-4.2, 0.0), "54-Cr": (4.55, 0.0),
    "55-Mn": (-3.75, 0.0),
    "54-Fe": (4.2, 0.0), "56-Fe": (9.94, 0.0), "57-Fe": (2.3, 0.0), "58-Fe": (15.0, 0.0),
    "59-Co": (2.49, 0.0),
    "58-Ni": (14.4, 0.0), "60-Ni": (2.8, 0.0), "61-Ni": (7.6, 0.0), "62-Ni": (-8.7, 0.0), "64-Ni": (-0.37, 0.0),
    "63-Cu": (6.43, 0.0), "65-Cu": (10.61, 0.0),
    "64-Zn": (5.22, 0.0), "66-Zn": (5.97, 0.0), "67-Zn": (7.56, 0.0), "68-Zn": (6.03, 0.0), "70-Zn": (6.0, 0.0),
    "69-Ga": (7.88, 0.0), "71-Ga": (6.4, 0.0),
    "70-Ge": (10.0, 0.0), "72-Ge": (8.51, 0.0), "73-Ge": (5.02, 0.0), "74-Ge": (7.58, 0.0), "76-Ge": (8.21, 0.0),
    "75-As": (6.58, 0.0),
    "74-Se": (0.8, 0.0), "76-Se": (12.2, 0.0), "77-Se": (8.25, 0.0), "78-Se": (8.24, 0.0), "80-Se": (7.48, 0.0), "82-Se": (6.34, 0.0),
    "79-Br": (6.8, 0.0), "81-Br": (6.79, 0.0),
    "86-Kr": (8.1, 0.0),
    "85-Rb": (7.03, 0.0), "87-Rb": (7.23, 0.0),
    "84-Sr": (7.0, 0.0), "86-Sr": (5.67, 0.0), "87-Sr": (7.4, 0.0), "88-Sr": (7.15, 0.0),
    "89-Y": (7.75, 0.0),
    "90-Zr": (6.4, 0.0), "91-Zr": (8.7, 0.0), "92-Zr": (7.4, 0.0), "94-Zr": (8.2, 0.0), "96-Zr": (5.5, 0.0),
    "93-Nb": (7.054, 0.0),
    "92-Mo": (6.91, 0.0), "94-Mo": (6.8, 0.0), "95-Mo": (6.91, 0.0), "96-Mo": (6.2, 0.0), "97-Mo": (7.24, 0.0), "98-Mo": (6.58, 0.0), "100-Mo": (6.73, 0.0),
    "99-Tc": (6.8, 0.0),
    "96-Ru": (0.0, 0.0), "98-Ru": (0.0, 0.0), "99-Ru": (6.9, 0.0), "100-Ru": (0.0, 0.0), "101-Ru": (3.3, 0.0), "102-Ru": (0.0, 0.0), "104-Ru": (0.0, 0.0),
    "103-Rh": (5.88, 0.0),
    "102-Pd": (7.7, 0.0), "104-Pd": (7.7, 0.0), "105-Pd": (5.5, 0.0), "106-Pd": (6.4, 0.0), "108-Pd": (4.1, 0.0), "110-Pd": (7.7, 0.0),
    "107-Ag": (7.555, 0.0), "109-Ag": (4.165, 0.0),
    "106-Cd": (5.0, 0.0), "108-Cd": (5.4, 0.0), "110-Cd": (5.9, 0.0), "111-Cd": (6.5, 0.0), "112-Cd": (6.4, 0.0), "113-Cd": (-8.0, -5.73), "114-Cd": (7.5, 0.0), "116-Cd": (6.3, 0.0),
    "113-In": (5.39, 0.0), "115-In": (4.01, -0.0562),
    "112-Sn": (6.1, 0.0), "114-Sn": (6.2, 0.0), "115-Sn": (6.0, 0.0), "116-Sn": (5.93, 0.0), "117-Sn": (6.48, 0.0), "118-Sn": (6.07, 0.0), "119-Sn": (6.12, 0.0), "120-Sn": (6.49, 0.0), "122-Sn": (5.74, 0.0), "124-Sn": (5.97, 0.0),
    "121-Sb": (5.71, 0.0), "123-Sb": (5.38, 0.0),
    "120-Te": (5.3, 0.0), "122-Te": (3.8, 0.0), "123-Te": (-0.05, -0.116), "124-Te": (7.96, 0.0), "125-Te": (5.02, 0.0), "126-Te": (5.56, 0.0), "128-Te": (5.89, 0.0), "130-Te": (6.02, 0.0),
    "127-I": (5.28, 0.0),
    "133-Cs": (5.42, 0.0),
    "130-Ba": (-3.6, 0.0), "132-Ba": (7.8, 0.0), "134-Ba": (5.7, 0.0), "135-Ba": (4.67, 0.0), "136-Ba": (4.91, 0.0), "137-Ba": (6.83, 0.0), "138-Ba": (4.84, 0.0),
    "138-La": (8.0, 0.0), "139-La": (8.24, 0.0),
    "136-Ce": (5.8, 0.0), "138-Ce": (6.7, 0.0), "140-Ce": (4.84, 0.0), "142-Ce": (4.75, 0.0),
    "141-Pr": (4.58, 0.0),
    "142-Nd": (7.7, 0.0), "143-Nd": (14.2, 0.0), "144-Nd": (2.8, 0.0), "145-Nd": (14.2, 0.0), "146-Nd": (8.7, 0.0), "148-Nd": (5.7, 0.0), "150-Nd": (5.3, 0.0),
    "147-Pm": (12.6, 0.0),
    "144-Sm": (-3.0, 0.0), "147-Sm": (14.0, 0.0), "148-Sm": (-3.0, 0.0), "149-Sm": (-19.2, -11.7), "150-Sm": (14.0, 0.0), "152-Sm": (-5.0, 0.0), "154-Sm": (9.3, 0.0),
    "151-Eu": (6.13, -2.53), "153-Eu": (8.22, 0.0),
    "152-Gd": (10.0, 0.0), "154-Gd": (10.0, 0.0), "155-Gd": (6.0, -17.0), "156-Gd": (6.3, 0.0), "157-Gd": (-1.14, -71.9), "158-Gd": (9.0, 0.0), "160-Gd": (9.15, 0.0),
    "159-Tb": (7.38, 0.0),
    "156-Dy": (6.1, 0.0), "158-Dy": (6.0, 0.0), "160-Dy": (6.7, 0.0), "161-Dy": (10.3, 0.0), "162-Dy": (-1.4, 0.0), "163-Dy": (5.0, 0.0), "164-Dy": (49.4, -0.79),
    "165-Ho": (8.01, 0.0),
    "162-Er": (8.8, 0.0), "164-Er": (8.2, 0.0), "166-Er": (10.6, 0.0), "167-Er": (3.0, 0.0), "168-Er": (7.4, 0.0), "170-Er": (9.6, 0.0),
    "169-Tm": (7.07, 0.0),
    "168-Yb": (-4.07, -0.62), "170-Yb": (6.77, 0.0), "171-Yb": (9.66, 0.0), "172-Yb": (9.43, 0.0), "173-Yb": (9.56, 0.0), "174-Yb": (19.3, 0.0), "176-Yb": (8.72, 0.0),
    "175-Lu": (7.24, 0.0), "176-Lu": (6.1, -0.57),
    "174-Hf": (10.9, 0.0), "176-Hf": (6.61, 0.0), "177-Hf": (0.8, 0.0), "178-Hf": (5.9, 0.0), "179-Hf": (7.46, 0.0), "180-Hf": (13.2, 0.0),
    "180-Ta": (7.0, 0.0), "181-Ta": (6.91, 0.0),
    "180-W": (5.0, 0.0), "182-W": (6.97, 0.0), "183-W": (6.53, 0.0), "184-W": (7.48, 0.0), "186-W": (-0.72, 0.0),
    "185-Re": (9.0, 0.0), "187-Re": (9.3, 0.0),
    "184-Os": (10.0, 0.0), "186-Os": (11.6, 0.0), "187-Os": (10.0, 0.0), "188-Os": (7.6, 0.0), "189-Os": (10.7, 0.0), "190-Os": (11.0, 0.0), "192-Os": (11.5, 0.0),
    "190-Pt": (9.0, 0.0), "192-Pt": (9.9, 0.0), "194-Pt": (10.55, 0.0), "195-Pt": (8.83, 0.0), "196-Pt": (9.89, 0.0), "198-Pt": (7.8, 0.0),
    "197-Au": (7.63, 0.0),
    "196-Hg": (30.3, 0.0), "199-Hg": (16.9, 0.0),
    "203-Tl": (6.99, 0.0), "205-Tl": (9.52, 0.0),
    "204-Pb": (9.9, 0.0), "206-Pb": (9.22, 0.0), "207-Pb": (9.28, 0.0), "208-Pb": (9.5, 0.0),
    "209-Bi": (8.532, 0.0),
    "226-Ra": (10.0, 0.0),
    "232-Th": (10.31, 0.0),
    "231-Pa": (9.1, 0.0),
    "233-U": (10.1, 0.0), "234-U": (12.4, 0.0), "235-U": (10.47, 0.0), "238-U": (8.402, 0.0),
    "237-Np": (10.55, 0.0),
    "239-Pu": (7.7, 0.0), "240-Pu": (3.5, 0.0), "242-Pu": (8.1, 0.0),
    "243-Am": (8.3, 0.0),
    "246-Cm": (9.3, 0.0), "248-Cm": (7.7, 0.0)
}

# 同位素别名
ISOTOPE_SYNONYMS = {
    "D": "2-H",
    "T": "3-H"
}

x, y, z = symbols('x y z')

def sanitize_filename(name: str) -> str:
    """清洗 Windows 文件名非法字符。"""
    bad = '<>:"/\\|?*'
    out = ''.join(('_' if ch in bad else ch) for ch in name)
    out = out.replace(' ', '')
    # 长度与重复点处理
    return out.strip().strip('.')

def format_formula_from_counts(counts: dict[str, float]) -> Tuple[str, List[str]]:
    """将元素计数化为最简整数比化学式，同时返回元素列表。"""
    if not counts:
        return "Unknown", []
    # gcd on scaled integers
    vals = list(counts.values())
    scale = 10**6
    ints = [max(1, int(round(v*scale))) for v in vals]
    g = ints[0]
    for t in ints[1:]:
        g = math.gcd(g, t)
    ratio = {el: round((cnt*scale)/g) for el, cnt in counts.items()}
    # 去除 1.0 的小数表示
    parts = []
    for el in sorted(ratio.keys()):  # 简单字母序
        n = int(ratio[el])
        parts.append(f"{el}{'' if n==1 else n}")
    formula = ''.join(parts)
    return formula, sorted(ratio.keys())

def get_metric_tensor(a, b, c, alpha, beta, gamma):
    alpha = np.deg2rad(alpha); beta = np.deg2rad(beta); gamma = np.deg2rad(gamma)
    G = np.zeros((3, 3))
    G[0, 0] = a ** 2; G[1, 1] = b ** 2; G[2, 2] = c ** 2
    G[0, 1] = G[1, 0] = a * b * np.cos(gamma)
    G[0, 2] = G[2, 0] = a * c * np.cos(beta)
    G[1, 2] = G[2, 1] = b * c * np.cos(alpha)
    return G

x, y, z = symbols('x y z')
def parse_symop(op_str):
    op_str = op_str.strip().strip("'\"")
    parts = op_str.split(',')
    if len(parts) != 3:
        raise ValueError("Invalid symmetry operation string")
    R = np.zeros((3, 3)); t = np.zeros(3)
    for i, p in enumerate(parts):
        expr = parse_expr(p.strip())
        const = float(expr.subs({x: 0, y: 0, z: 0}))
        t[i] = const % 1
        for var, j in zip([x, y, z], range(3)):
            coeff = float(expr.coeff(var) or 0)
            R[i, j] = coeff
    return R, t

def pseudo_voigt(x, x0, fwhm, eta):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2))); gamma = fwhm / 2
    return eta * (gamma**2 / ((x - x0)**2 + gamma**2)) / np.pi + \
           (1 - eta) * np.exp(-(x - x0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def resolve_b(label: str) -> complex:
    key = ISOTOPE_SYNONYMS.get(label, label)
    if key in ISOTOPE_SCATTERING_LENGTHS:
        re, im = ISOTOPE_SCATTERING_LENGTHS[key]; return complex(re, im)
    if key in NATURAL_SCATTERING_LENGTHS:
        re, im = NATURAL_SCATTERING_LENGTHS[key]; return complex(re, im)
    raise ValueError(f"未找到元素/同位素 {label} 的中子散射长度。")

def _elem_from_site(site: dict):
    import re
    el = site.get('element')
    if isinstance(el, str) and el.strip().lower() in ('biso','uiso','u_iso','b_iso','u_iso_or_equiv','b_iso_or_equiv'):
        el = None
    if el and isinstance(el, str):
        m = re.match(r'([A-Z][a-z]?)', el.strip())
        if m: return m.group(1)
    lbl = site.get('label') or site.get('Label') or ''
    m = re.match(r'([A-Z][a-z]?)', str(lbl))
    return m.group(1) if m else 'X'

def build_atoms_from_cif(cif_path: Path):
    text = cif_path.read_text(encoding='utf-8', errors='ignore')
    data, loops = parse_cif_text(text)
    cell = get_cell_params(data)
    a = float(cell.get("a")); b = float(cell.get("b")); c = float(cell.get("c"))
    alpha = float(cell.get("alpha")); beta = float(cell.get("beta")); gamma = float(cell.get("gamma"))
    G = get_metric_tensor(a, b, c, alpha, beta, gamma); G_star = np.linalg.inv(G)
    sym_ops_str = get_symmetry_operations(data, loops) or ["x,y,z"]
    sym_ops = [parse_symop(op) for op in sym_ops_str]

    # point group rotations
    point_group_R = []
    for R, _ in sym_ops:
        if not any(np.allclose(R, ex, atol=1e-4) for ex in point_group_R):
            point_group_R.append(R)

    # atoms
    atom_sites_cif = get_atom_sites(data, loops, text) or []
    asym_atoms = []
    for site in atom_sites_cif:
        fx, fy, fz = site.get('fract_x'), site.get('fract_y'), site.get('fract_z')
        if fx is None or fy is None or fz is None: continue
        occ = site.get('occupancy', 1.0) if site.get('occupancy') is not None else 1.0
        biso = site.get('Biso')
        if biso is None and site.get('Uiso') is not None:
            biso = 8.0 * np.pi**2 * float(site.get('Uiso'))
        if biso is None: biso = 0.5
        elem = _elem_from_site(site)
        asym_atoms.append({
            "label": elem,
            "pos": np.array([fx, fy, fz], dtype=float),
            "occ": float(occ),
            "B_iso": float(biso),
            "site_label": site.get('label')
        })

    # expand by symmetry
    all_atoms = []
    atom_sites = defaultdict(list)
    tol = 1e-4
    for idx, atom in enumerate(asym_atoms):
        try:
            b_val = resolve_b(atom["label"]).real
        except Exception:
            # 跳过未知元素
            continue
        equiv_pos = []
        for R, t in sym_ops:
            new_pos = np.mod(np.dot(R, atom['pos']) + t, 1)
            if not any(np.allclose(new_pos, ex, atol=tol) for ex in equiv_pos):
                equiv_pos.append(new_pos)
        atom_sites[idx] = equiv_pos
        for pos in equiv_pos:
            all_atoms.append({
                'label': atom['label'],
                'pos': pos,
                'b': b_val,
                'occ': atom['occ'],
                'b_iso': atom.get('B_iso', 0.5),
                'site_idx': idx
            })

    # counts for simplest ratio
    element_count = defaultdict(float)
    for idx, equiv_pos in atom_sites.items():
        if idx >= len(asym_atoms): continue
        atom = asym_atoms[idx]
        mult = len(equiv_pos)
        element_count[atom['label']] += mult * atom['occ']

    # space group name
    sg = get_space_group(data) or 'Unknown'

    return {
        'cell': (a, b, c, alpha, beta, gamma),
        'G_star': G_star,
        'point_group_R': point_group_R,
        'atoms_all': all_atoms,
        'element_count': dict(element_count),
        'space_group': sg
    }

def simulate_pattern(G_star, point_group_R, all_atoms, lam, tth_min, tth_max, fwhm=0.05, eta=0.5):
    # hmax estimate
    a = math.sqrt(1/G_star[0,0]) if G_star[0,0] > 0 else 1.0
    b = math.sqrt(1/G_star[1,1]) if G_star[1,1] > 0 else 1.0
    c = math.sqrt(1/G_star[2,2]) if G_star[2,2] > 0 else 1.0
    sin_theta_max = np.sin(np.deg2rad(tth_max / 2))
    min_d = lam / (2 * sin_theta_max) if sin_theta_max > 0 else 1e-6
    hmax = int(max(a, b, c) / min_d) + 2

    peaks = []
    seen_orbits = set()
    for h in range(0, hmax + 1):
        start_k = -hmax if h > 0 else 0
        for k in range(start_k, hmax + 1):
            start_l = -hmax if (h > 0 or k > 0) else 1
            for l in range(start_l, hmax + 1):
                if h == 0 and k == 0 and l == 0: continue
                hkl = np.array([h, k, l])
                q2 = np.dot(hkl, np.dot(G_star, hkl))
                if q2 <= 0: continue
                d = 1 / np.sqrt(q2)
                sin_theta = lam / (2 * d)
                if sin_theta > 1 or sin_theta <= 0: continue
                two_theta = 2 * np.rad2deg(np.arcsin(sin_theta))
                if two_theta < tth_min or two_theta > tth_max: continue
                F = 0 + 0j
                for atom in all_atoms:
                    dw = np.exp(-atom['b_iso'] * (sin_theta ** 2) / (lam ** 2))
                    phase = np.exp(2j * np.pi * np.dot(hkl, atom['pos']))
                    F += atom['b'] * atom['occ'] * phase * dw
                intens = np.abs(F) ** 2
                if intens < 1e-3: continue
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
                lorentz = 1 / (np.sin(np.deg2rad(two_theta)) * np.sin(np.deg2rad(two_theta / 2)))
                I = m * intens * lorentz
                peaks.append((two_theta, I, (h, k, l), m))

    peaks.sort(key=lambda x: x[0])

    # 生成展宽光谱
    x_grid = np.linspace(tth_min, tth_max, 2000)
    y_grid = np.zeros_like(x_grid)
    for tt, I, _, _ in peaks:
        y_grid += I * pseudo_voigt(x_grid, tt, fwhm, eta)
    return peaks, x_grid, y_grid, hmax

def save_outputs(out_base: Path, base_name: str, formula: str, sg: str, x: np.ndarray, y: np.ndarray, peaks, elements: List[str]):
    out_dir_tif = out_base / 'tif'
    out_dir_txt = out_base / 'txt'
    out_dir_tif.mkdir(parents=True, exist_ok=True)
    out_dir_txt.mkdir(parents=True, exist_ok=True)

    # 使用 CIF 文件名作为输出基名（保留空格，替换非法字符）
    bad = '<>:"/\\|?*'
    name = ''.join(('_' if ch in bad else ch) for ch in base_name).strip().strip('.')

    tif_path = out_dir_tif / f"{name}.tif"
    txt_path = out_dir_txt / f"{name}.txt"

    # 防重名
    idx = 1
    while tif_path.exists() or txt_path.exists():
        tif_path = out_dir_tif / f"{name}__{idx}.tif"
        txt_path = out_dir_txt / f"{name}__{idx}.txt"
        idx += 1

    # 保存图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, lw=1.0, color='C0')
    ax.set_xlabel('2θ (degrees)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'{formula} | {sg}')
    fig.tight_layout()
    fig.savefig(tif_path, dpi=300, format='tif')
    plt.close(fig)

    # 保存峰表
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"最简比化学式: {formula}\n")
        f.write(f"空间群: {sg}\n")
        f.write(f"包含原子: {', '.join(elements)}\n")
        f.write("衍射峰位与峰强(2θ_deg, Intensity, h, k, l, multiplicity):\n")
        for tt, I, (h, k, l), m in peaks:
            f.write(f"{tt:.4f}\t{I:.6f}\t{h}\t{k}\t{l}\t{m}\n")

    return tif_path, txt_path

def run_batch(input_path: str, out_dir: str, lam: float, tth_min: float, tth_max: float, fwhm: float, eta: float):
    paths = list_cif_files(input_path) if Path(input_path).exists() else []
    if not paths and Path(input_path).is_file():
        paths = [Path(input_path)]
    if not paths:
        print(f"未找到 CIF 文件: {input_path}")
        return

    out_base = Path(out_dir)
    ok = 0; fail = 0
    for i, p in enumerate(paths, 1):
        try:
            info = build_atoms_from_cif(p)
            G_star = info['G_star']
            peaks, x, y, hmax = simulate_pattern(G_star, info['point_group_R'], info['atoms_all'], lam, tth_min, tth_max, fwhm, eta)
            formula, elements = format_formula_from_counts(info['element_count'])
            sg = info['space_group'] or 'Unknown'
            # 使用 CIF 文件名（不含扩展名）作为输出文件名
            save_outputs(out_base, p.stem, formula, sg, x, y, peaks, elements)
            ok += 1
            print(f"[{i}/{len(paths)}] 完成: {p.name} -> {p.stem}.tif / {p.stem}.txt")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(paths)}] 失败: {p} | {e}")
    print(f"完成。成功 {ok} 个，失败 {fail} 个。输出目录: {Path(out_dir).resolve()}")

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="批量生成中子粉末模拟谱图（CIF -> tif + txt）")
    ap.add_argument('input', help='CIF 文件或目录（递归搜索）')
    ap.add_argument('-o', '--out', default='outputs', help='输出根目录（默认: outputs）')
    ap.add_argument('--lam', type=float, default=1.5405, help='中子波长 Å（默认 1.5405）')
    ap.add_argument('--tth', type=float, nargs=2, metavar=('MIN', 'MAX'), default=(10.0, 120.0), help='2θ 范围（度）')
    ap.add_argument('--fwhm', type=float, default=0.05, help='展宽 FWHM（默认 0.05°）')
    ap.add_argument('--eta', type=float, default=0.5, help='Pseudo-Voigt 混合因子（默认 0.5）')
    return ap.parse_args(argv)

# 新增：Tk 对话框导入（用于弹框选择路径）
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None

# 新增：弹框选择输入与输出路径
def ask_paths_via_gui() -> tuple[str | None, str | None]:
    if filedialog is None:
        return None, None
    root = tk.Tk()
    root.withdraw()
    # 先询问选择目录还是单文件
    choose_dir = True
    if messagebox:
        choose_dir = messagebox.askyesno(
            title="选择输入",
            message="是否选择一个包含 CIF 的目录？\n是: 目录\n否: 单个 CIF 文件"
        )
    # 输入路径
    if choose_dir:
        in_path = filedialog.askdirectory(
            title="选择包含 CIF 文件的目录",
            initialdir=str(Path.cwd())
        )
    else:
        in_path = filedialog.askopenfilename(
            title="选择 CIF 文件",
            initialdir=str(Path.cwd()),
            filetypes=[("CIF files", "*.cif;*.CIF"), ("All files", "*.*")]
        )
    if not in_path:
        root.destroy()
        return None, None
    # 输出目录
    out_dir = filedialog.askdirectory(
        title="选择输出目录",
        initialdir=str(Path.cwd())
    )
    root.destroy()
    if not out_dir:
        return None, None
    return in_path, out_dir

def main():
    # 优先使用弹框选择
    if filedialog is not None:
        in_path, out_dir = ask_paths_via_gui()
        if not in_path or not out_dir:
            print("已取消：未选择输入或输出路径。")
            return
        # 使用默认参数
        lam = 1.5405
        tth_min, tth_max = 10.0, 120.0
        fwhm, eta = 0.05, 0.5
        run_batch(in_path, out_dir, lam, tth_min, tth_max, fwhm, eta)
        return

    # 兜底：无 Tk 时走命令行参数
    args = parse_args()
    run_batch(args.input, args.out, args.lam, args.tth[0], args.tth[1], args.fwhm, args.eta)

if __name__ == '__main__':
    main()