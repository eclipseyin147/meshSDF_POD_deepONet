import os
import numpy as np
import scipy.io as io
import torch
import time
"""
Data Merging and Preprocessing Pipeline

This script implements an adaptation of the E_flow preprocessing and 
merging logic originally developed by @xie-lab-ml.
Source: https://github.com/xie-lab-ml/multiphysics-bench/main/FNO/merge_data_TE_heat.py
"""
start_time = time.time()

mat_dir_train = 'training/E_flow/'
output_file_train = 'merge_data/E_flow_train_128.pt'
os.makedirs(os.path.dirname(output_file_train), exist_ok=True)

mat_dir_test = "testing/E_flow/"
output_file_test = 'merge_data/E_flow_test_128.pt'
os.makedirs(os.path.dirname(output_file_test), exist_ok=True)

#  train
num_samples_train = 10000
shape = (128, 128)

x_data = np.zeros((num_samples_train, *shape), dtype=np.float32)
y_data = np.zeros((num_samples_train, 3, *shape), dtype=np.float32)



# input kappa
# load max_kappa  min_kappa
range_allkappa_paths = f"{mat_dir_train}kappa/range_allkappa.mat"
range_allkappa = io.loadmat(range_allkappa_paths)['range_allkappa']

max_kappa = range_allkappa[0,1]
min_kappa = range_allkappa[0,0]

# output  ec_V, u_flow, v_flow
# load max_ec_V min_ec_V
range_allec_V_paths = f"{mat_dir_train}ec_V/range_allec_V.mat"
range_allec_V = io.loadmat(range_allec_V_paths)['range_allec_V']

max_ec_V = range_allec_V[0,1]
min_ec_V = range_allec_V[0,0]

# load max_u_flow min_u_flow
range_allu_flow_paths = f"{mat_dir_train}u_flow/range_allu_flow.mat"
range_allu_flow = io.loadmat(range_allu_flow_paths)['range_allu_flow']

max_u_flow = range_allu_flow[0,1]
min_u_flow = range_allu_flow[0,0]


# load max_v_flow min_v_flow
range_allv_flow_paths = f"{mat_dir_train}v_flow/range_allv_flow.mat"
range_allv_flow = io.loadmat(range_allv_flow_paths)['range_allv_flow']

max_v_flow = range_allv_flow[0,1]
min_v_flow = range_allv_flow[0,0]


for idx in range(num_samples_train):
    # kappa
    path_kappa = os.path.join(f"{mat_dir_train}kappa/", f'{idx+1}.mat')
    kappa = io.loadmat(path_kappa)['export_kappa']
    kappa_normalized = (kappa - min_kappa) / (max_kappa - min_kappa) * 1.8 - 0.9 # [-0.9,0.9]

    x_data[idx] = kappa_normalized


    # ec_V
    path_ec_V = os.path.join(f"{mat_dir_train}ec_V/", f'{idx+1}.mat')
    ec_V = io.loadmat(path_ec_V)['export_ec_V']
    ec_V_normalized = (ec_V - min_ec_V) / (max_ec_V - min_ec_V) * 1.8 - 0.9 # [-0.9,0.9]

    # u_flow
    path_u_flow = os.path.join(f"{mat_dir_train}u_flow/", f'{idx+1}.mat')
    u_flow = io.loadmat(path_u_flow)['export_u_flow']
    u_flow_normalized = (u_flow - min_u_flow) / (max_u_flow - min_u_flow) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 0] = ec_V_normalized.astype(np.float32)
    y_data[idx, 1] = u_flow_normalized.astype(np.float32)


    # v_flow
    path_v_flow = os.path.join(f"{mat_dir_train}v_flow/", f'{idx+1}.mat')
    v_flow = io.loadmat(path_v_flow)['export_v_flow']
    v_flow_normalized = (v_flow - min_v_flow) / (max_v_flow - min_v_flow) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = v_flow_normalized.astype(np.float32)


    if idx % 200 == 0:
        print("train: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
        print("train: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())


# 转换为Pyv_floworch张量
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
y_data = np.zeros((num_samples_test, 3, *shape), dtype=np.float32)


for idx in range(num_samples_test):
    # kappa 
    path_kappa = os.path.join(f"{mat_dir_test}kappa/", f'{idx+10001}.mat')
    kappa = io.loadmat(path_kappa)['export_kappa']
    kappa_normalized = (kappa - min_kappa) / (max_kappa - min_kappa) * 1.8 - 0.9 # [-0.9,0.9]

    x_data[idx] = kappa_normalized


    # ec_V
    path_ec_V = os.path.join(f"{mat_dir_test}ec_V/", f'{idx+10001}.mat')
    ec_V = io.loadmat(path_ec_V)['export_ec_V']
    ec_V_normalized = (ec_V - min_ec_V) / (max_ec_V - min_ec_V) * 1.8 - 0.9 # [-0.9,0.9]

    # u_flow
    path_u_flow = os.path.join(f"{mat_dir_test}u_flow/", f'{idx+10001}.mat')
    u_flow = io.loadmat(path_u_flow)['export_u_flow']
    u_flow_normalized = (u_flow - min_u_flow) / (max_u_flow - min_u_flow) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 0] = ec_V_normalized.astype(np.float32)
    y_data[idx, 1] = u_flow_normalized.astype(np.float32)


    # v_flow 
    path_v_flow = os.path.join(f"{mat_dir_test}v_flow/", f'{idx+10001}.mat')
    v_flow = io.loadmat(path_v_flow)['export_v_flow']
    v_flow_normalized = (v_flow - min_v_flow) / (max_v_flow - min_v_flow) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = v_flow_normalized.astype(np.float32)

    if idx % 200 == 0:
        print("test: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
        print("test: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())


# 转换为Pyv_floworch张量
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




