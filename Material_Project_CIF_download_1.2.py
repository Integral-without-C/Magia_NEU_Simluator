import os
import pandas as pd
from mp_api.client import MPRester
import re
from pymatgen.core import Structure

api_key = 'e62QN94J9vzWgw3HhISZBFurST9Tm5NQ'
cif_dir = 'F:\\MP_data\\cif_files'
os.makedirs(cif_dir, exist_ok=True)

data_list = []
failed_list = []

def safe_filename(s):
    # 替换非法字符
    return re.sub(r'[\\/:*?"<>|\s]', '_', str(s))

with MPRester(api_key) as mpr:
    mp_docs = mpr.materials.summary.search(
        elements=["Li"],
        fields=["material_id", "database_IDs", "formula_pretty", "symmetry", "deprecated", "theoretical"],
        chunk_size=1000,  # 最大限制为1000
        num_chunks=10     # 处理10个数据块，共计10000条记录，在这里修改总数目
    )

    for mp_doc in mp_docs:
        mpid = str(mp_doc.material_id)
        formula = mp_doc.formula_pretty
        database_IDs = mp_doc.database_IDs
        crystal_system = getattr(mp_doc.symmetry, "crystal_system", "Unknown")
        space_group = getattr(mp_doc.symmetry, "symbol", "Unknown")
        is_high_quality = not mp_doc.deprecated
        is_theoretical = mp_doc.theoretical

        # 自动命名
        cif_filename = f"{safe_filename(mpid)}+{safe_filename(formula)}+{safe_filename(crystal_system)}+{safe_filename(space_group)}.cif"
        cif_path = os.path.join(cif_dir, cif_filename)

        # 下载带对称性的CIF文件（symmetrized），失败重试一次
        success = False
        for attempt in range(2):
            try:
                structure: Structure = mpr.get_structure_by_material_id(mpid, conventional_unit_cell=True)
                structure.to(filename=cif_path, fmt="cif", symprec=0.1)
                success = True
                break
            except Exception as e:
                if attempt == 1:
                    print(f"{mpid} CIF下载失败: {e}")
                    cif_path = "下载失败"
                    failed_list.append({
                        "MP_ID": mpid,
                        "Formula": formula,
                        "Crystal_System": crystal_system,
                        "Space_Group": space_group,
                        "Error": str(e)
                    })

        icsd_ids = database_IDs.get("icsd", []) if database_IDs and "icsd" in database_IDs else [""]
        for icsd_id in icsd_ids:
            data_list.append([
                icsd_id, mpid, formula, crystal_system, space_group, is_high_quality, is_theoretical, cif_path
            ])

icsd_to_mpid_df = pd.DataFrame(
    data_list,
    columns=['ICSD_ID', 'MP_ID', 'Formula', 'Crystal_System', 'Space_Group', 'Is_High_Quality', 'Is_Theoretical', 'CIF_File']
)

output_file_path = 'F:\\MP_data\\icsd_to_mpid_with_formula_crystal_system_space_group_quality_theory_and_cif.xlsx'
icsd_to_mpid_df.to_excel(output_file_path, index=False)

# 输出下载失败详细信息
if failed_list:
    failed_file = os.path.join(cif_dir, "download_failed.txt")
    with open(failed_file, "w", encoding="utf-8") as f:
        for item in failed_list:
            f.write(f"{item['MP_ID']}\t{item['Formula']}\t{item['Crystal_System']}\t{item['Space_Group']}\t{item['Error']}\n")
    print(f"下载失败详细信息已保存到 {failed_file}")

print(f'已将信息输出到 {output_file_path}')
print(f'CIF 文件保存在 {cif_dir}')