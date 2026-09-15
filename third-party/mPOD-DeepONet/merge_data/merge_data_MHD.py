import os
import numpy as np
import scipy.io as io
import torch
import time
import time
"""
Data Merging and Preprocessing Pipeline

This script implements an adaptation of the MHD preprocessing and 
merging logic originally developed by @xie-lab-ml.
Source: https://github.com/xie-lab-ml/multiphysics-bench/main/FNO/merge_data_TE_heat.py
"""

start_time = time.time()

mat_dir_train = 'training/MHD/'
output_file_train = 'merge_data/MHD_train_128.pt'
os.makedirs(os.path.dirname(output_file_train), exist_ok=True)

mat_dir_test = "testing/MHD/"
output_file_test = 'merge_data/MHD_test_128.pt'
os.makedirs(os.path.dirname(output_file_test), exist_ok=True)

#  train
num_samples_train = 10000
shape = (128, 128)

x_data = np.zeros((num_samples_train, *shape), dtype=np.float32)
y_data = np.zeros((num_samples_train, 5, *shape), dtype=np.float32)



# input Br
# load max_Br  min_Br
range_allBr_paths = f"{mat_dir_train}Br/range_allBr.mat"
range_allBr = io.loadmat(range_allBr_paths)['range_allBr']

max_Br = range_allBr[0,1]
min_Br = range_allBr[0,0]

# output  Jx, Jy, Jz, u_u, u_v
# load max_Jx min_Jx
range_allJx_paths = f"{mat_dir_train}Jx/range_allJx.mat"
range_allJx = io.loadmat(range_allJx_paths)['range_allJx']

max_Jx = range_allJx[0,1]
min_Jx = range_allJx[0,0]

# load max_Jy min_Jy
range_allJy_paths = f"{mat_dir_train}Jy/range_allJy.mat"
range_allJy = io.loadmat(range_allJy_paths)['range_allJy']

max_Jy = range_allJy[0,1]
min_Jy = range_allJy[0,0]

# load max_Jz min_Jz
range_allJz_paths = f"{mat_dir_train}Jz/range_allJz.mat"
range_allJz = io.loadmat(range_allJz_paths)['range_allJz']

max_Jz = range_allJz[0,1]
min_Jz = range_allJz[0,0]

# load max_u_u min_u_u
range_allu_u_paths = f"{mat_dir_train}u_u/range_allu_u.mat"
range_allu_u = io.loadmat(range_allu_u_paths)['range_allu_u']

max_u_u = range_allu_u[0,1]
min_u_u = range_allu_u[0,0]


# load max_u_v min_u_v
range_allu_v_paths = f"{mat_dir_train}u_v/range_allu_v.mat"
range_allu_v = io.loadmat(range_allu_v_paths)['range_allu_v']

max_u_v = range_allu_v[0,1]
min_u_v = range_allu_v[0,0]


for idx in range(num_samples_train):
    # Br
    path_Br = os.path.join(f"{mat_dir_train}Br/", f'{idx+1}.mat')
    Br = io.loadmat(path_Br)['export_Br']
    Br_normalized = (Br - min_Br) / (max_Br - min_Br) * 1.8 - 0.9 # [-0.9,0.9]

    x_data[idx] = Br_normalized


    # Jx
    path_Jx = os.path.join(f"{mat_dir_train}Jx/", f'{idx+1}.mat')
    Jx = io.loadmat(path_Jx)['export_Jx']
    Jx_normalized = (Jx - min_Jx) / (max_Jx - min_Jx) * 1.8 - 0.9 # [-0.9,0.9]

    # Jy
    path_Jy = os.path.join(f"{mat_dir_train}Jy/", f'{idx+1}.mat')
    Jy = io.loadmat(path_Jy)['export_Jy']
    Jy_normalized = (Jy - min_Jy) / (max_Jy - min_Jy) * 1.8 - 0.9 # [-0.9,0.9]

    # Jz
    path_Jz = os.path.join(f"{mat_dir_train}Jz/", f'{idx+1}.mat')
    Jz = io.loadmat(path_Jz)['export_Jz']
    Jz_normalized = (Jz - min_Jz) / (max_Jz - min_Jz) * 1.8 - 0.9 # [-0.9,0.9]

    # u_u
    path_u_u = os.path.join(f"{mat_dir_train}u_u/", f'{idx+1}.mat')
    u_u = io.loadmat(path_u_u)['export_u']
    u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]

    # u_v
    path_u_v = os.path.join(f"{mat_dir_train}u_v/", f'{idx+1}.mat')
    u_v = io.loadmat(path_u_v)['export_v']
    u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 0] = Jx_normalized.astype(np.float32)
    y_data[idx, 1] = Jy_normalized.astype(np.float32)
    y_data[idx, 2] = Jz_normalized.astype(np.float32)
    y_data[idx, 3] = u_u_normalized.astype(np.float32)
    y_data[idx, 4] = u_v_normalized.astype(np.float32)


    if idx % 200 == 0:
        print("train: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
        print("train: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())


# 转换为Pyu_vorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

if len(x_tensor.shape) == 3:
    x_tensor = x_tensor.unsqueeze(1)
    print('reshape x:', x_tensor.shape)

# 保存为pt文件
torch.save({'x': x_tensor, 'y': y_tensor}, output_file_train)

print(f"数据已成功保存为 {output_file_train}")
print(f"输入形状: {x_tensor.shape}")
print(f"输出形状: {y_tensor.shape}")

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")




#  test  归一化

num_samples_test = 1000        
shape = (128, 128)       


x_data = np.zeros((num_samples_test, *shape), dtype=np.float32)
y_data = np.zeros((num_samples_test, 5, *shape), dtype=np.float32)


for idx in range(num_samples_test):
    # Br 
    path_Br = os.path.join(f"{mat_dir_test}Br/", f'{idx+10001}.mat')
    Br = io.loadmat(path_Br)['export_Br']
    Br_normalized = (Br - min_Br) / (max_Br - min_Br) * 1.8 - 0.9 # [-0.9,0.9]

    x_data[idx] = Br_normalized


    # Jx
    path_Jx = os.path.join(f"{mat_dir_test}Jx/", f'{idx+10001}.mat')
    Jx = io.loadmat(path_Jx)['export_Jx']
    Jx_normalized = (Jx - min_Jx) / (max_Jx - min_Jx) * 1.8 - 0.9 # [-0.9,0.9]

    # Jy
    path_Jy = os.path.join(f"{mat_dir_test}Jy/", f'{idx+10001}.mat')
    Jy = io.loadmat(path_Jy)['export_Jy']
    Jy_normalized = (Jy - min_Jy) / (max_Jy - min_Jy) * 1.8 - 0.9 # [-0.9,0.9]

    # Jz
    path_Jz = os.path.join(f"{mat_dir_test}Jz/", f'{idx+10001}.mat')
    Jz = io.loadmat(path_Jz)['export_Jz']
    Jz_normalized = (Jz - min_Jz) / (max_Jz - min_Jz) * 1.8 - 0.9 # [-0.9,0.9]

    # u_u
    path_u_u = os.path.join(f"{mat_dir_test}u_u/", f'{idx+10001}.mat')
    u_u = io.loadmat(path_u_u)['export_u']
    u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]

    # u_v 
    path_u_v = os.path.join(f"{mat_dir_test}u_v/", f'{idx+10001}.mat')
    u_v = io.loadmat(path_u_v)['export_v']
    u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 0] = Jx_normalized.astype(np.float32)
    y_data[idx, 1] = Jy_normalized.astype(np.float32)
    y_data[idx, 2] = Jz_normalized.astype(np.float32)
    y_data[idx, 3] = u_u_normalized.astype(np.float32)
    y_data[idx, 4] = u_v_normalized.astype(np.float32)

    if idx % 200 == 0:
        print("test: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
        print("test: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())


# 转换为Pyu_vorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

# print('x shape:', x_tensor.shape)
if len(x_tensor.shape) == 3:
    x_tensor = x_tensor.unsqueeze(1)
    print('reshape x:', x_tensor.shape)

# 保存为pt文件
torch.save({'x': x_tensor, 'y': y_tensor}, output_file_test)

print(f"数据已成功保存为 {output_file_test}")
print(f"输入形状: {x_tensor.shape}")
print(f"输出形状: {y_tensor.shape}")

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    




