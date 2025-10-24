#!/usr/bin/env python3
"""
cif_reader.py

轻量级 CIF 解析器 —— 提取并打印：
 1) 化学组成（优先使用 CIF 中的 _chemical_formula_* 字段，否则从 atom_site 计算）
 2) 空间群
 3) 晶胞参数 a, b, c, alpha, beta, gamma
 4) 所有原子的信息（坐标、占有率 occupancy、Biso 或 Uiso）

用法：
    python cif_reader.py <path-to-cif>

依赖：仅 Python 标准库 + pandas （可选，用于 Tabular 显示）

说明：此脚本对常见 FullProf / CIF 输出格式进行容错处理（包括 Uiso 标注在列中或作为列名）。
"""

from __future__ import annotations
import re
import sys
from collections import defaultdict
from pathlib import Path
import json

try:
    import pandas as pd
except Exception:
    pd = None

# 新增：tkinter 文件选择对话框（可在没有命令行参数时使用）
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

def ask_cif_file_dialog(initialdir: str = '.'):
    """打开文件选择对话框，返回所选文件路径或 None"""
    if filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    fp = filedialog.askopenfilename(title='选择 CIF 文件', initialdir=initialdir,
                                    filetypes=[('CIF files', '*.cif;*.CIF'), ('All files', '*.*')])
    root.destroy()
    return fp


def parse_cif_text(text: str):
    lines = [ln.rstrip() for ln in text.splitlines()]
    data = {}
    loops = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.lower().startswith("data_"):
            data["_data_block"] = ln
            i += 1
            continue
        if ln.startswith("loop_"):
            i += 1
            keys = []
            # collect keys
            while i < n and lines[i].strip().startswith("_"):
                keys.append(lines[i].strip())
                i += 1
            # collect rows until next blank/loop/_/data_
            rows_of_tokens = []
            while i < n and lines[i].strip() != "" and not lines[i].strip().startswith("loop_") and not lines[i].strip().startswith("_") and not lines[i].strip().lower().startswith("data_"):
                rowline = lines[i].strip()
                # skip comment lines
                if rowline.startswith('#'):
                    i += 1
                    continue
                
                # 修正的 token 提取逻辑：
                # 1. 查找所有带引号的字符串或不含空格的连续字符。
                # 2. 对每个匹配项，去除首尾的引号。
                parts = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", rowline)
                cleaned_parts = []
                for p in parts:
                    if (p.startswith("'") and p.endswith("'")) or \
                       (p.startswith('"') and p.endswith('"')):
                        cleaned_parts.append(p[1:-1])
                    else:
                        cleaned_parts.append(p)
                
                rows_of_tokens.append(cleaned_parts)
                i += 1

            if len(keys) > 0 and rows_of_tokens:
                # 如果每行的 token 数与 key 的数量不匹配，则可能是旧的扁平化解析方式
                if len(rows_of_tokens[0]) != len(keys):
                    flat_tokens = [token for row in rows_of_tokens for token in row]
                    per = len(keys)
                    if len(flat_tokens) % per == 0:
                         rows = [flat_tokens[j:j+per] for j in range(0, len(flat_tokens), per)]
                         loops.append({"keys": keys, "rows": rows})
                else: # 正常情况，每行 token 数等于 key 的数量
                    loops.append({"keys": keys, "rows": rows_of_tokens})
            continue
        # key-value
        if ln.startswith("_"):
            parts = ln.split(None, 1)
            key = parts[0]
            if len(parts) > 1:
                val = parts[1]
            else:
                j = i+1
                val = ""
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n:
                    val = lines[j].strip()
                    i = j
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]
            data[key] = val
            i += 1
            continue
        i += 1
    return data, loops


def extract_atom_sites_from_loops_loose(loops):
    # prefer loops that contain _atom_site_label
    for loop in loops:
        keys = loop['keys']
        lowkeys = [k.lower() for k in keys]
        if any(k.startswith('_atom_site_') for k in lowkeys) and any(k == '_atom_site_label' for k in lowkeys):
            rows = loop['rows']
            if rows:
                # 使用小写键的索引映射，兼容不同字符大小写与命名变体
                mapping = {k: idx for idx, k in enumerate(lowkeys)}
                atom_list = []
                for r in rows:
                    atom = {}
                    def get_by_lk(lk):
                        idx = mapping.get(lk)
                        return r[idx] if (idx is not None and idx < len(r)) else None
                    # 常见列名变体处理
                    atom['label'] = get_by_lk('_atom_site_label')
                    # type / element 列可能用 _atom_site_type_symbol 或 _atom_site_type_symbol 等
                    # 搜索包含 'type_symbol' 的列
                    elem_idx = None
                    for lk in mapping.keys():
                        if 'type_symbol' in lk or lk.endswith('type') or lk.endswith('symbol'):
                            elem_idx = mapping[lk]; break
                    atom['element'] = r[elem_idx] if (elem_idx is not None and elem_idx < len(r)) else get_by_lk('_atom_site_type_symbol')
                    # fractional coordinates
                    atom['fract_x'] = get_by_lk('_atom_site_fract_x')
                    atom['fract_y'] = get_by_lk('_atom_site_fract_y')
                    atom['fract_z'] = get_by_lk('_atom_site_fract_z')
                    # occupancy / Biso / Uiso
                    atom['occupancy'] = get_by_lk('_atom_site_occupancy') or get_by_lk('_atom_site_occupancy') 
                    atom['Biso'] = get_by_lk('_atom_site_b_iso_or_equiv') or get_by_lk('_atom_site_B_iso_or_equiv') or get_by_lk('_atom_site_b_iso')
                    atom['Uiso'] = get_by_lk('_atom_site_u_iso_or_equiv') or get_by_lk('_atom_site_U_iso_or_equiv') or get_by_lk('_atom_site_u_iso')
                    atom['adp_type'] = get_by_lk('_atom_site_adp_type')
                    # wyckoff / multiplicity（T2 CIF 中存在）
                    wyck_idx = None
                    mult_idx = None
                    for lk in mapping.keys():
                        if 'wyckoff' in lk:
                            wyck_idx = mapping[lk]
                        if 'multiplicity' in lk:
                            mult_idx = mapping[lk]
                    atom['wyckoff'] = r[wyck_idx] if (wyck_idx is not None and wyck_idx < len(r)) else None
                    atom['multiplicity'] = (int(r[mult_idx]) if (mult_idx is not None and mult_idx < len(r) and re.match(r'^\d+$', str(r[mult_idx]))) else None)
                    atom_list.append(atom)
                # convert numeric strings to floats where possible
                def tofloat(s):
                    if s is None: return None
                    mo = re.match(r"([+-]?[0-9]*\.?[0-9]+)", str(s))
                    return float(mo.group(1)) if mo else None
                for a in atom_list:
                    for k in ('fract_x','fract_y','fract_z','occupancy','Biso','Uiso'):
                        if a.get(k) is not None:
                            a[k] = tofloat(a[k])
                return atom_list
    # fallback: heuristic parsing when rows are free-format following the loop definition
    return None


def heuristic_extract_from_text(text: str):
    # locate the loop block that lists atom_site keys, then collect subsequent non-comment, non-empty lines as rows
    lines = text.splitlines()
    for idx, ln in enumerate(lines):
        if ln.strip().lower().startswith('loop_'):
            look = [lines[j].strip() for j in range(idx+1, min(idx+41, len(lines)))]
            if any(l.lower().startswith('_atom_site_label') for l in look):
                # find last key index
                last_key_rel = None
                keys = [l for l in look if l.startswith('_atom_site_')]
                for rel_i, l in enumerate(look):
                    if l.startswith('_atom_site_'):
                        last_key_rel = rel_i
                data_start_idx = idx + 1 + (last_key_rel + 1 if last_key_rel is not None else 0)
                rows = []
                for j in range(data_start_idx, len(lines)):
                    s = lines[j].strip()
                    if s == '':
                        # stop if subsequent non-empty looks like new section
                        ahead = None
                        for k in range(j+1, min(j+6, len(lines))):
                            if lines[k].strip()!='':
                                ahead = lines[k].strip()
                                break
                        if ahead is None or ahead.lower().startswith('loop_') or ahead.startswith('_') or ahead.startswith('#'):
                            break
                        else:
                            continue
                    if s.lower().startswith('loop_') or s.startswith('_') or s.lower().startswith('data_'):
                        break
                    if s.startswith('#'):
                        continue
                    rows.append(s)
                # parse rows heuristically
                parsed = []
                for rr in rows:
                    # try simple split first
                    tokens = rr.split()
                    parsed.append(tokens)
                atom_sites = []
                for tokens in parsed:
                    atom = {'label':None,'element':None,'fract_x':None,'fract_y':None,'fract_z':None,'occupancy':None,'Biso':None,'Uiso':None,'adp_type':None,'wyckoff':None,'multiplicity':None}
                    # 支持两类常见行格式：
                    # (A) label elem mult Wyckoff x y z Biso occ  -> 如 T2 CIF: Ni1 Ni2+ 4 a 0 0 0 . 1.
                    # (B) label x y z rest -> 如另一类格式
                    if len(tokens) >= 7 and re.match(r'^\d+$', tokens[2]) and re.match(r'^[A-Za-z]$', tokens[3]):
                        atom['label'] = tokens[0]
                        atom['element'] = tokens[1]
                        atom['multiplicity'] = int(tokens[2])
                        atom['wyckoff'] = tokens[3]
                        atom['fract_x'] = tokens[4]; atom['fract_y'] = tokens[5]; atom['fract_z'] = tokens[6]
                        rest = tokens[7:]
                        if len(rest) >= 1: atom['Biso'] = rest[0]
                        if len(rest) >= 2: atom['occupancy'] = rest[1]
                    elif len(tokens) >= 4:
                        atom['label']=tokens[0]; atom['fract_x']=tokens[1]; atom['fract_y']=tokens[2]; atom['fract_z']=tokens[3]
                        rest = tokens[4:]
                        # try to detect adp token in rest
                        adp_idx = None
                        for i,t in enumerate(rest):
                            if t.lower() in ('uiso','biso','u_iso','b_iso','u_iso_or_equiv','b_iso_or_equiv'):
                                adp_idx = i
                                break
                        if adp_idx is not None:
                            atom['adp_type'] = rest[adp_idx]
                            prev = rest[:adp_idx]
                            nums = [p for p in prev if re.match(r"[+-]?[0-9]*\.?[0-9]+", p)]
                            if len(nums) >= 1: atom['Biso'] = nums[0]
                            if len(nums) >= 2: atom['occupancy'] = nums[1]
                            if len(rest) > adp_idx+1: atom['element'] = rest[adp_idx+1]
                        else:
                            if len(rest) >= 1: atom['element'] = rest[-1]
                            if len(rest) >= 2: atom['Biso'] = rest[-2]
                            if len(rest) >= 3: atom['occupancy'] = rest[-3]
                    atom_sites.append(atom)
                # convert numeric strings to floats where possible
                def tofloat(s):
                    if s is None: return None
                    mo = re.match(r"([+-]?[0-9]*\.?[0-9]+)", str(s))
                    return float(mo.group(1)) if mo else None
                for a in atom_sites:
                    for k in ('fract_x','fract_y','fract_z','occupancy','Biso','Uiso'):
                        if a.get(k) is not None:
                            a[k] = tofloat(a[k])
                return atom_sites
    return []


def compute_composition_from_sites(atom_sites):
    comp = defaultdict(float)
    for a in atom_sites:
        el = a.get('element') or ''
        m = re.match(r'([A-Za-z]+)', str(el))
        if m:
            symbol = m.group(1)
        else:
            lbl = a.get('label','')
            m2 = re.match(r'([A-Za-z]+)', str(lbl))
            symbol = m2.group(1) if m2 else el
        occ = a.get('occupancy')
        try:
            occf = float(occ) if occ is not None else 1.0
        except:
            mo = re.match(r'([0-9.+-Ee]+)', str(occ))
            occf = float(mo.group(1)) if mo else 1.0
        comp[symbol] += occf
    return dict(sorted(comp.items(), key=lambda x:x[0]))


def get_chemical_formula(data: dict, atom_sites: list | None = None) -> str | None:
    """从 data 或 atom_sites 中获取化学式（优先 data 中的字段）。"""
    for k in ('_chemical_formula_sum', '_chemical_formula_moiety', '_chemical_formula_structural'):
        if k in data:
            return data[k]
    if atom_sites:
        comp = compute_composition_from_sites(atom_sites)
        return ' '.join(f"{el}{(int(v) if abs(v-round(v))<1e-6 else v)}" for el,v in comp.items())
    return None


def get_space_group(data: dict) -> str | None:
    """从 data 字典中检索空间群相关字段（多种常见前缀）。"""
    for kd in data.keys():
        kl = kd.lower()
        if kl.startswith('_space_group') or kl.startswith('_symmetry_space_group') or kl.startswith('_symmetry_int_tables_number'):
            return data[kd]
    return None


def get_cell_params(data: dict) -> dict:
    """从 data 中提取晶胞参数 a,b,c,alpha,beta,gamma（返回 dict，缺失为 None）。"""
    cell = {}
    keys_map = {
        'a': '_cell_length_a',
        'b': '_cell_length_b',
        'c': '_cell_length_c',
        'alpha': '_cell_angle_alpha',
        'beta': '_cell_angle_beta',
        'gamma': '_cell_angle_gamma'
    }
    for name, k in keys_map.items():
        found = None
        for kd in data.keys():
            if kd.lower() == k.lower():
                found = data[kd]; break
        if found is None:
            cell[name] = None
        else:
            mo = re.match(r"([+-]?[0-9]*\.?[0-9]+)", str(found))
            cell[name] = float(mo.group(1)) if mo else found
    return cell


def get_atom_sites(data: dict, loops: list, text: str) -> list:
    """返回 atom_sites 列表：优先从 loops 解析，否则启发式从文本解析。"""
    atom_sites = extract_atom_sites_from_loops_loose(loops)
    if atom_sites is None or len(atom_sites) == 0:
        atom_sites = heuristic_extract_from_text(text)
    return atom_sites


def get_symmetry_operations(data: dict, loops: list) -> list:
    """
    从 CIF 数据中读取显式列出的对称性操作。

    优先选择 operation 列（如 _space_group_symop_operation_xyz 或 _symmetry_equiv_pos_as_xyz），
    并避免误选 *_id 列。
    """
    ops = []

    # 优先的精确列名
    preferred_exact = [
        '_space_group_symop_operation_xyz',
        '_symmetry_equiv_pos_as_xyz',
    ]
    # 兜底的模糊匹配片段（排除 *_id）
    fallback_contains = [
        'operation_xyz',
        'pos_as_xyz',
    ]

    for loop in loops:
        keys = loop.get('keys', [])
        rows = loop.get('rows', [])
        if not keys or not rows:
            continue

        lowkeys = [k.lower() for k in keys]

        # 1) 精确匹配优先
        op_idx = -1
        for exact in preferred_exact:
            if exact in lowkeys:
                op_idx = lowkeys.index(exact)
                break

        # 2) 兜底：包含匹配，但显式排除 *_id
        if op_idx == -1:
            for i, lk in enumerate(lowkeys):
                if any(tok in lk for tok in fallback_contains) and not lk.endswith('_id'):
                    op_idx = i
                    break

        # 3) 再兜底：任意包含 symop 且包含 operation，且不是 *_id
        if op_idx == -1:
            for i, lk in enumerate(lowkeys):
                if '_symop_' in lk and 'operation' in lk and not lk.endswith('_id'):
                    op_idx = i
                    break

        if op_idx != -1:
            for r in rows:
                if op_idx < len(r):
                    op_str = str(r[op_idx]).strip().strip("'\"")
                    if op_str and op_str not in ops:
                        ops.append(op_str)
            if ops:
                return ops

    return ops


def main(path):
    p = Path(path)
    if not p.exists():
        print('ERROR: file not found', path); return
    text = p.read_text(encoding='utf-8',errors='ignore')
    data, loops = parse_cif_text(text)

    # atom sites first (so composition can use sites if needed)
    atom_sites = get_atom_sites(data, loops, text)

    # formula / composition
    composition = get_chemical_formula(data, atom_sites)

    # space group
    sg = get_space_group(data)

    # cell
    cell = get_cell_params(data)

    # 新增：读取对称性操作
    symmetry_ops = get_symmetry_operations(data, loops)

    # output
    out = {
        'chemical_formula': composition,
        'space_group': sg,
        'cell': cell,
        'n_atom_sites': len(atom_sites) if atom_sites is not None else 0,
        'n_symmetry_ops': len(symmetry_ops), # 新增
        'atom_sites': atom_sites,
    }
    print(json.dumps({k:v for k,v in out.items() if k!='atom_sites'}, indent=2))
    
    # 新增：打印对称性操作
    if symmetry_ops:
        print('\nSymmetry operations:')
        for s in symmetry_ops:
            print('  ', s)

    if pd is not None and atom_sites:
        df = pd.DataFrame(atom_sites)
        print('\nAtom sites table:')
        print(df.to_string(index=False))
    else:
        # print brief atom site list
        if atom_sites:
            print('\nAtom sites (list):')
            for a in atom_sites:
                print('  ', a)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 如果未提供命令行参数，尝试弹出文件选择对话框
        selected = ask_cif_file_dialog(initialdir=str(Path.cwd()))
        if not selected:
            print('Usage: python cif_reader.py <cif-file>')
        else:
            main(selected)
    else:
        main(sys.argv[1])
