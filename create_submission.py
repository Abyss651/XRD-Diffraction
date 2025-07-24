import torch
from pymatgen.core import Structure, Lattice, Element
from tqdm import tqdm
import pandas as pd

data = torch.load("/personal/competition_diffcsp/DiffCSP/mpts_52_ckpt/eval_diff.pt", map_location='cpu')
frac_coords = data['frac_coords']  # Tensor形状[1, 27820, 3]
num_atoms = data['num_atoms']      # Tensor形状[1, 2000]
atom_types = data['atom_types']    # Tensor形状[1, 27820]
lattices = data['lattices']        # Tensor形状[1, 2000, 3, 3]

# 假设输入数据为以下变量
# frac_coords: Tensor形状[1, 27820, 3]
# num_atoms: Tensor形状[1, 2000]
# atom_types: Tensor形状[1, 27820]
# lattices: Tensor形状[1, 2000, 3, 3]

# 去除批次维度
frac_coords = frac_coords.squeeze(0)  # [27820, 3]
num_atoms = num_atoms.squeeze(0)      # [2000]
atom_types = atom_types.squeeze(0)     # [27820]
lattices = lattices.squeeze(0)        # [2000, 3, 3]

# 确保张量在CPU上以便转换为numpy
frac_coords = frac_coords.cpu()
num_atoms = num_atoms.cpu().long()    # 转换为整型
atom_types = atom_types.cpu()
lattices = lattices.cpu()

# 验证原子总数是否匹配
total_atoms = num_atoms.sum().item()
assert total_atoms == frac_coords.shape[0], "原子总数不匹配"
assert len(atom_types) == total_atoms, "原子类型数量不匹配"

# 拆分每个晶体的数据
structures = []
current_idx = 0
for i in range(num_atoms.shape[0]):
    n_i = num_atoms[i].item()
    # 提取第i个晶体的数据
    coords = frac_coords[current_idx:current_idx + n_i].numpy()
    types = atom_types[current_idx:current_idx + n_i].numpy().astype(int)
    lattice_matrix = lattices[i].numpy()
    
    # 转换为元素对象列表
    species = [Element.from_Z(z) for z in types]
    
    # 创建Lattice和Structure
    lattice = Lattice(lattice_matrix)
    structure = Structure(lattice, species, coords)
    structures.append(structure)
    
    current_idx += n_i

# 最终的结构列表包含2000个Structure对象
print(f"成功生成{len(structures)}个晶体结构")

# 生成submission文件
header = ["ID", "cif"]
rows = []
# store cif files and create a list of dictionary for spgroup number and crystal system
for i in tqdm(range(len(structures)), desc="Generating CIF files"):
    # check if id is a str or numpy.int
    ID = f"A-{i+1}"
    cif = structures[i].to(fmt="cif")
    rows.append([ID, cif])
# save header and rows to a csv file
df = pd.DataFrame(rows, columns=header)
df.to_csv(f"submission.csv", index=False)

