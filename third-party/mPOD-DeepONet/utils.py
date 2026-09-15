import gc
import torch
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.delta_init = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, epoch):
        # if epoch > 500 and self.delta_init == self.min_delta:
        #     self.min_delta = self.min_delta*1e-1
        # if (validation_loss) < self.min_validation_loss:
        #     self.min_validation_loss = validation_loss
        #     self.counter = 0
        # elif (validation_loss) > (self.min_validation_loss + self.min_delta):
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         return True
        return False
    
class ChannelwiseMSE(nn.Module):
    def __init__(self, dim_channel=1):
        super().__init__()
        self.dim_channel=dim_channel
    def forward(self, true, pred, reduce=True):
        loss = (true-pred).square()
        reduce_dims = list(range(true.ndim))
        reduce_dims.pop(self.dim_channel)
        loss = loss.mean(dim=reduce_dims)
        if reduce:
            return loss.sum()
        else:
            return loss

class Learner():
    def __init__(self, model, 
                 epochs=1000,*, 
                 device="cpu", model_name="model",
                 optim="Adam", patience=10, min_delta=1e-3):
        self.model, self.model_name = model, model_name
        self.device = device

        if optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif optim == 'AdamW2':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-5, betas=(0.95, 0.99))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-9)
        elif optim == 'AdamWwarm':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-5)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2, eta_min=1e-8)
        elif optim == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0000, amsgrad=False)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=.8)
        elif optim == 'AdamCos':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0000, amsgrad=False)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=epochs)
        # else:
        #     self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=.5)
        self.epochs = epochs
        self.start_epoch = 0
        self.optim = optim
        self.early_stopper = EarlyStopper(patience=patience, min_delta=min_delta) 
        self.mse = ChannelwiseMSE() # reduction='sum'
    
    def model_summarize(self, only_sum = True):
        psum = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"total number of parameters: {psum:,}")

    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f'Model size: {size_all_mb:.3f} MB')
        
    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.optim == "AdamW":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif 'Adam' in self.optim:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=.8)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=.5)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logs = {key:checkpoint['logs'][key] for key in ['train_time', 'test_time','train_loss', 'test_loss', 'grad_norm']}
        
        self.start_epoch = checkpoint.get('epoch', 0)
        print(f"Successfully loaded checkpoint from epoch {self.start_epoch}")
        self.start_epoch +=1
        
    def train(self, trainloader, testloader, epochs, *, 
              print_every=10, debug=False, 
              save=True, save_every=True):
    
        self.trainloader, self.testloader = trainloader, testloader
        checkpoint = {'train_time':[], 'test_time':[],'train_loss':[], 'test_loss':[], 'grad_norm':[]}
        if self.start_epoch > 0:
            checkpoint = {key:self.logs[key] for key in ['train_time', 'test_time','train_loss', 'test_loss', 'grad_norm']}
        total_epochs = self.start_epoch+epochs
        for epoch in range(self.start_epoch, total_epochs):
            self.model.train()
            # Use Python lists to store epoch-specific values as floats
            epoch_train_losses = 0 # []
            epoch_grad_norms = 0 # []
            
            start_e = time.time()
            
            # Progress bar logic
            loader = tqdm(self.trainloader) if debug else self.trainloader
            
            train_size = 0
            batch_count = 0
            for i, train_batch in enumerate(loader):

                loss_val, grad_norm_val = self.train_step(train_batch)

                bsize = train_batch['x'].size(0)
                epoch_train_losses += loss_val*bsize
                epoch_grad_norms += grad_norm_val
                train_size += bsize
                batch_count += 1
                
            time_e = time.time() - start_e
            
            # Calculate averages on CPU
            avg_train_loss = epoch_train_losses / train_size
            avg_grad_norm = epoch_grad_norms / batch_count
            if 'Adam' == self.optim:
                self.scheduler.step(avg_train_loss)
            else:
                self.scheduler.step()
            
            # Append to checkpoint lists
            checkpoint['train_time'].append(time_e)
            checkpoint['train_loss'].append(avg_train_loss)
            checkpoint['grad_norm'].append(avg_grad_norm)

            torch.cuda.empty_cache()
            # Evaluation phase
            if (epoch % print_every == 0) or (epoch == total_epochs - 1):
                self.model.eval()
                epoch_test_losses = 0
                test_size = 0
                start_test = time.time()
                
                with torch.no_grad():
                    for test_batch in self.testloader:
                        test_loss_val = self.test_step(test_batch)
                        bsize = test_batch['x'].size(0)
                        epoch_test_losses += test_loss_val*bsize
                        test_size += bsize
                
                avg_test_loss = epoch_test_losses / test_size
                time_test = time.time() - start_test
                
                print(f"epoch:{epoch} | time: {time_e:.2f}/{time_test:.2f} | loss: {avg_train_loss:.2e} | test: {avg_test_loss:.2e} | norm: {avg_grad_norm:.2f}")
                
                checkpoint['test_loss'].append(avg_test_loss)
                checkpoint['test_time'].append(time_test)
                
                if save_every:
                    self.save_model(self.model_name, epoch, total_epochs, self.model, self.optimizer, self.scheduler, checkpoint)
                
                if self.early_stopper.early_stop(avg_test_loss, epoch):
                    print(f"Early stop at epoch:{epoch}!")
                    break
        if save:
            self.save_model(self.model_name, epoch, total_epochs, self.model, self.optimizer, self.scheduler, checkpoint)

        print(f"Finish epoch:{epoch} | train loss: {avg_train_loss:.4e} | test loss: {avg_test_loss:.4e}")
        
        return checkpoint
    
    def save_model(self, model_name, epoch, total_epochs, model, optimizer, scheduler, logs):
        # timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"logs/{model_name}_epoch={total_epochs}.pt" # _{timestr}
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'logs':logs
            }, fname)

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        input = (batch['x']).to(self.device).float()
        true = (batch['y']).to(self.device).float()
        pred = self.model(input)
        loss = self.mse(pred, true)
        loss.backward()
        total_loss_val = loss.item()  

        grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float('inf'),
                norm_type='inf')
        self.optimizer.step()
        return total_loss_val , grad_norm.item()
    
    @torch.no_grad()
    def test_step(self, batch):
        true = (batch['y']).to(self.device).float()
        input = (batch['x']).to(self.device).float()
        pred = self.model(input)
        loss = self.mse(pred, true)
        return loss.item()
    
    @torch.no_grad()
    def eval_channel(self):
        epoch_test_losses = 0
        test_size = 0        
        with torch.no_grad():
            for batch in self.testloader:
                true = (batch['y']).to(self.device).float()
                input = (batch['x']).to(self.device).float()
                pred = self.model(input)
                test_loss_val = self.mse(pred, true, reduce=False)
                bsize = batch['x'].size(0)
                epoch_test_losses += test_loss_val*bsize
                test_size += bsize
        
        avg_test_loss = epoch_test_losses / test_size
        print('Channel Loss: ', end = '')
        for i in range(len(avg_test_loss)):
            print(f'var{i}: {avg_test_loss[i]:.4e}', end=' ')
        print('\n')

        return avg_test_loss
       
        # print(f"epoch:{epoch} | time: {time_e:.2f}/{time_test:.2f} | loss: {avg_train_loss:.2e} | test: {avg_test_loss:.2e} | norm: {avg_grad_norm:.2f}") 


def OutNormalizer(y_train):
    from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
    # channel-wise
    reduce_dims = list(range(y_train.ndim))
    reduce_dims.pop(1)
    output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
    output_encoder.fit(y_train)
    return output_encoder

class Normalizer:
    def __init__(self, y_train, scale=10):
        # We use .detach() to ensure these are constants, not part of the graph
        self.y_min = torch.min(y_train).detach()
        self.y_max = torch.max(y_train).detach()
        self.scale = scale # Fixed: uses the scale passed in __init__

    def encode(self, y):
        # Maps data from [min, max] to [0, scale]
        return (y - self.y_min) / (self.y_max - self.y_min) * self.scale

    def decode(self, y_norm):
        # Maps data from [0, scale] back to original range
        return self.y_min + y_norm * (self.y_max - self.y_min) / self.scale

class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.x = data_dict['x']
        self.y = data_dict['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a dictionary instead of a tuple
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "id": idx  # You can even pass metadata!
        }

class MPData:
    def __init__(self, data_path, name, dry_run=True, 
             normalized_y=True, *, 
             n_train=None, n_test=None, batch_size=32):
        train_data = torch.load(f"{data_path}/{name}_train_128.pt")
        test_data = torch.load(f"{data_path}/{name}_test_128.pt")
        if 'TE_heat' in name:
            n_train = 9000 if n_train is None else n_train
            n_test = 1000 if n_test is None else n_test 
        else:
            n_train = 10000 if n_train is None else n_train
            n_test = 1000 if n_test is None else n_test

        if dry_run:
            batch_size = 10 if batch_size > 10 else batch_size
            n_train, n_test = 100, batch_size
        
        self.n_train, self.n_test, self.batch_size = n_train, n_test, batch_size
        
        for data, size in zip([train_data, test_data], [n_train, n_test]):
            data['x'] = data['x'][:size]
            data['y'] = data['y'][:size]

        if normalized_y:
            out_normalizer = OutNormalizer(train_data['y'])
            train_data['y'] = out_normalizer(train_data['y'])
            test_data['y'] = out_normalizer(test_data['y'])
            self.out_normalizer = out_normalizer
        
        self.train_data = train_data
        self.test_data = test_data
        

    def loader(self):
        train_loader = DataLoader(
            DictDataset(self.train_data), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            DictDataset(self.test_data), batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader

import matplotlib.pyplot as plt
def plot_data(dataloader, datatype, 
              transform_y=None, transform_x=None, 
              pred_model=None, device='cpu',*, channel_x = 1, channel_y=1, channel=None,  name_dict=['x', 'y'],
              idx=0, complex_out=False, y_range=None, figsize=None):
    
    # 1. Fetch Batch and Slices
    batch = next(iter(dataloader))
    bx, by = batch[name_dict[0]], batch[name_dict[1]]

    if channel is not None:
        channel_x, channel_y = channel, channel
    
    if transform_x:
        bx = transform_x(bx)
    
    x_slice = bx[idx]
    y_raw_slice = by[idx]
    
    # 2. Shape Correction Helper (Works on sliced data)
    def fix_shape(tensor, name, channel):
        """Ensures tensor is (Channel, Height, Width) after slice"""
        if tensor.ndim == 3: # Handle (H, W, C) -> (C, H, W)
            if tensor.shape[-1] == channel and tensor.shape[0] != channel:
                tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 2: # Handle (H, W) -> (1, H, W)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 1: # Handle flattened (Res*Res)
            res = int(np.sqrt(tensor.numel() // channel))
            tensor = tensor.reshape(channel, res, res)
            
        assert tensor.shape[0] == channel, f"Shape mismatch for {name}: expected {channel} channels, got {tensor.shape}"
        return tensor

    x_slice = fix_shape(x_slice, "Input X", channel_x)
    y_raw_slice = fix_shape(y_raw_slice, "Raw Y", channel_y)

    # 3. Determine Plotting Mode and Data
    # Column configuration: [Col 0, Col 1, Col 2]
    if pred_model:
        # Mode: Input X | Target Y (Transformed) | Prediction
        pred_model.to(device).eval()
        with torch.no_grad():
            preds = pred_model(bx.to(device))
            y_pred_slice = fix_shape(preds[idx], "Prediction", channel_y)
            if complex_out:
                y_pred_slice = torch.fft.irfft2(torch.complex(y_pred_slice[0], y_pred_slice[1])).unsqueeze(0)
            y_pred_slice = y_pred_slice.detach().cpu()
        
        y_target_slice = fix_shape(transform_y(by)[idx], "Transformed Y", channel_y) if transform_y else y_raw_slice

        error_map = y_target_slice - y_pred_slice
        
        # Fourier Error Spectrum calculation
        ffterror = torch.fft.fft2(error_map)
        ffterror = torch.fft.fftshift(ffterror)
        ffterror = torch.fft.fftn(error_map, dim=(-2, -1)) # Ensure 2D FFT on spatial dims
        ffterror = torch.fft.fftshift(ffterror, dim=(-2, -1))
        error_spec = (ffterror.abs() + 1).log10()

        plot_data_list = [x_slice, y_target_slice, y_pred_slice, error_map, error_spec]
        titles = ['Input X', 'Target Y', 'Pred Y', 'Error (Spatial)', 'Error Spectrum']
        num_cols = 5

    elif transform_y:
        # Mode: Input X | Raw Y | Transformed Y
        y_trans_slice = fix_shape(transform_y(by)[idx], "Transformed Y", channel_y)
        plot_data_list = [x_slice, y_raw_slice, y_trans_slice]
        titles = ['Input X', 'Raw Y', 'Transformed Y']
        num_cols = 3
    else:
        # Mode: Input X | Raw Y
        plot_data_list = [x_slice, y_raw_slice]
        titles = ['Input X', 'Raw Y']
        num_cols = 2

    # 4. Execute Plotting
    plot_channel = max(channel_x, channel_y)
    if figsize is None:
        figsize = (num_cols * 4, plot_channel * 3.5)
    
    fig, axes = plt.subplots(plot_channel, num_cols, figsize=figsize, squeeze=False)
    cmap_x = 'gray' if datatype == 'darcy' else 'viridis'

    for c in range(plot_channel):
        # Shared color scale for Y comparisons (Col 1 and Col 2)

        if num_cols >= 2:
            y_vals = [plot_data_list[1][c]]
            if num_cols == 3: y_vals.append(plot_data_list[2][c])
            v_lims = y_range if y_range else (min(v.min().item() for v in y_vals), max(v.max().item() for v in y_vals))

        for col in range(num_cols):
            if c >= channel_x and col == 0:
                axes[c, col].set_axis_off() 
                continue
            data = plot_data_list[col][c]
            ax = axes[c, col]
            
            # Formatting logic based on column content
            curr_title = f"{titles[col]}"
            curr_cmap = 'gray' if (col == 0 and datatype == 'darcy') else 'magma' if 'Spectrum' in titles[col] else 'viridis'
            
            # Use shared limits only for Target and Prediction columns
            if col in [1, 2] and num_cols >= 3:
                clim = v_lims
            elif 'Error (Spatial)' in curr_title:
                # Symmetric error map or centered at 0
                max_err = data.abs().max().item()
                clim = (-max_err, max_err)
                curr_cmap = 'bwr' # Red/Blue diverging map for errors
            elif 'Error Spectrum' in curr_title:
                curr_cmap = 'Blues_r'  # Reversed Blues for high-freq residuals
                clim = (data.min().item(), data.max().item())
            else:
                clim = (data.min().item(), data.max().item())

            _plot_to_ax(ax, data, curr_title, curr_cmap, clim)

    plt.tight_layout()
    plt.show()

def _plot_to_ax(ax, data, title, cmap, lims):
    im = ax.imshow(data.detach().cpu().numpy(), cmap=cmap, vmin=lims[0], vmax=lims[1], origin='lower')
    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)



    
    
