import os
import numpy as np
from scipy import io
import torch
import time
"""
Data Merging and Preprocessing Pipeline

Source: https://github.com/xie-lab-ml/multiphysics-bench/main/FNO/merge_data_TE_heat.py
Author: @xie-lab-ml

Key Modifications:
- split the training date with 9000/1000 since the original source from multiphysics-bench didn't provide test data for TE_heat.
"""
start_time = time.time()

#  train
mat_dir_train = 'training/TE_heat/' 
output_file_train = 'merge_data/TE_heat_train_128.pt'  
os.makedirs(os.path.dirname(output_file_train), exist_ok=True)

# [Errno 2] No such file or directory: 'testing/TE_heat/mater/10001.mat'
mat_dir_test = 'training/TE_heat/' 
output_file_test = 'merge_data/TE_heat_test_128.pt'  
os.makedirs(os.path.dirname(output_file_test), exist_ok=True)

num_samples = 9000         
shape = (128, 128)       


x_data = np.zeros((num_samples, *shape), dtype=np.float32)
y_data = np.zeros((num_samples, 3, *shape), dtype=np.float32)



for idx in range(num_samples):
    mater_path = os.path.join(mat_dir_train, f'mater/{idx+1}.mat')
    mater_data = io.loadmat(mater_path)['mater'].astype(np.float32)
    
    mater_in = (mater_data >= 1e11) & (mater_data <= 3e11)
    mater_out = (mater_data >= 10) & (mater_data <= 20)

    normal_datamater = np.where(mater_in, (mater_data - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater_data - 10) / (20 - 10) * 0.8 - 0.9)
    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

    x_data[idx] = normal_datamater


# output material  Re(Ez) Im(Ez) T
    # load max_abs_Ez
max_abs_Ez_path = f"{mat_dir_train}Ez/max_abs_Ez.mat"
max_abs_Ez = io.loadmat(max_abs_Ez_path)['max_abs_Ez']

print(max_abs_Ez)

# load T
range_allT_paths = f"{mat_dir_train}T/range_allT.mat"
range_allT = io.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


for idx in range(num_samples):

    Ez_path = os.path.join(mat_dir_train, f'Ez/{idx+1}.mat')
    Ez_data = io.loadmat(Ez_path)['export_Ez'].astype(np.complex64)
    
    Ez_normalized = Ez_data / max_abs_Ez * 0.9  # 保持相位不变  [0,0.9]
    real_Ez_normalized = np.real(Ez_normalized)
    imag_Ez_normalized = np.imag(Ez_normalized)

    y_data[idx, 0] = real_Ez_normalized.astype(np.float32)
    y_data[idx, 1] = imag_Ez_normalized.astype(np.float32)

    # T 
    T_path = os.path.join(mat_dir_train, f'T/{idx+1}.mat')
    T_data = io.loadmat(T_path)['export_T'].astype(np.float32)
    T_normalized = (T_data - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = T_normalized.astype(np.float32)

    if idx % 200 == 0:
        print(f"Saved combined array for index {idx} to {output_file_train }")
        print("Min:", y_data.min(), "Max:", y_data.max())

# 转换为PyTorch张量
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



num_samples = 1000        
shape = (128, 128)       


x_data = np.zeros((num_samples, *shape), dtype=np.float32)
y_data = np.zeros((num_samples, 3, *shape), dtype=np.float32)



for idx in range(num_samples):
    mater_path = os.path.join(mat_dir_test, f'mater/{idx+num_samples+1}.mat')
    mater_data = io.loadmat(mater_path)['mater'].astype(np.float32)
    
    mater_in = (mater_data >= 1e11) & (mater_data <= 3e11)
    mater_out = (mater_data >= 10) & (mater_data <= 20)

    normal_datamater = np.where(mater_in, (mater_data - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater_data - 10) / (20 - 10) * 0.8 - 0.9)
    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

    x_data[idx] = normal_datamater


# output material  Re(Ez) Im(Ez) T
    # load max_abs_Ez
max_abs_Ez_path = f"{mat_dir_train}Ez/max_abs_Ez.mat"
max_abs_Ez = io.loadmat(max_abs_Ez_path)['max_abs_Ez']

print(max_abs_Ez)

# load T
range_allT_paths = f"{mat_dir_train}T/range_allT.mat"
range_allT = io.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


for idx in range(num_samples):

    Ez_path = os.path.join(mat_dir_test, f'Ez/{idx+num_samples+1}.mat')
    Ez_data = io.loadmat(Ez_path)['export_Ez'].astype(np.complex64)
    
    Ez_normalized = Ez_data / max_abs_Ez * 0.9  # 保持相位不变  [0,0.9]
    real_Ez_normalized = np.real(Ez_normalized)
    imag_Ez_normalized = np.imag(Ez_normalized)

    y_data[idx, 0] = real_Ez_normalized.astype(np.float32)
    y_data[idx, 1] = imag_Ez_normalized.astype(np.float32)

    # T 
    T_path = os.path.join(mat_dir_test, f'T/{idx+num_samples+1}.mat')
    T_data = io.loadmat(T_path)['export_T'].astype(np.float32)
    T_normalized = (T_data - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = T_normalized.astype(np.float32)


    if idx % 200 == 0:
        print(f"Saved combined array for index {idx} to {output_file_test }")
        print("Min:", y_data.min(), "Max:", y_data.max())

# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

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




