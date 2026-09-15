import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ChannelwiseMSE(nn.Module):
    def __init__(self, dim_channel=1, dim_sample=0):
        super().__init__()
        self.dim_channel = dim_channel
        self.dim_sample = dim_sample
        self.spatial_dims = [d for d in range(4) if d not in (self.dim_channel, self.dim_sample)]
        
    def _spatial_sq_error(self, true, pred):
        """Returns MSE per sample and per channel: Shape (N, C)"""
        se = (true - pred).square()
        return se.mean(dim=self.spatial_dims)

    def mean(self, true, pred):
        """Global Mean: Mean of (Sum of channels) across samples"""
        # (N, C) -> Sum across C -> Mean across N
        mse_nc = self._spatial_sq_error(true, pred)
        return mse_nc.sum(dim=self.dim_channel).mean(dim=self.dim_sample)

    def sd(self, true, pred):
        """Global SD: Standard Deviation of (Sum of channels) across samples"""
        # (N, C) -> Sum across C -> Std across N
        mse_nc = self._spatial_sq_error(true, pred)
        return mse_nc.sum(dim=self.dim_channel).std(dim=self.dim_sample)

    def cmean(self, true, pred):
        """Channel-wise Mean: Mean across samples for each channel: Shape (C)"""
        # (N, C) -> Mean across N
        mse_nc = self._spatial_sq_error(true, pred)
        return mse_nc.mean(dim=self.dim_sample)

    def csd(self, true, pred):
        """Channel-wise SD: SD across samples for each channel: Shape (C)"""
        # (N, C) -> Std across N
        mse_nc = self._spatial_sq_error(true, pred)
        return mse_nc.std(dim=self.dim_sample)

'''
CISE table
'''
def fmt(value, scale=1e3):
    # This turns 0.000368 into 3.68
    v_scaled = value * scale
    return f"{v_scaled:.2f}"

# best p only
def evaluate_models_to_df(data_dict):
    criterion = ChannelwiseMSE(dim_channel=1, dim_sample=0)
    true = data_dict['y_tr']
    model_keys = [k for k in data_dict.keys() if k not in ['x', 'y_tr']]
    
    # --- PASS 1: Calculate Total CISE for Everyone ---
    screening_results = []
    
    for model_key in model_keys:
        pred = data_dict[model_key]
        with torch.no_grad():
            # Only compute the scalar Global Mean (cheaper than channel-wise)
            g_mean = criterion.mean(true, pred).item()
            g_sd = criterion.sd(true, pred).item()
            
        # Parse Name immediately to help with grouping
        # Checks if key has "(p)"; if not, p is NaN (e.g., FNO)
        if '(' in model_key:
            base_name = model_key.split('(')[0].strip()
            p_val = int(model_key.split('(')[1].strip(')'))
        else:
            base_name = model_key
            p_val = np.nan
            
        screening_results.append({
            "Model_Key": model_key,   # Keep the full key to retrieve tensor later
            "Model": base_name,
            "$p$": p_val,
            "Total Mean": g_mean,
            "Total SD": g_sd
        })
    
    df_screen = pd.DataFrame(screening_results)
    
    # --- SELECTION: Find the Best 'p' for each Model Architecture ---
    # idxmin gives the index of the row with the lowest Total Mean for each group
    best_indices = df_screen.groupby('Model')['Total Mean'].idxmin()
    best_df = df_screen.loc[best_indices].copy()
    
    print(f"Selected Best Models:\n{best_df[['Model_Key']]}")
    
    # --- PASS 2: Calculate Channel-wise Metrics ONLY for Winners ---
    # We iterate only through the rows in 'best_df'
    for idx, row in best_df.iterrows():
        model_key = row['Model_Key']
        pred = data_dict[model_key]
        
        with torch.no_grad():
            # Now we do the expensive channel-wise reduction
            c_means = criterion.cmean(true, pred).cpu().numpy()
            c_sds = criterion.csd(true, pred).cpu().numpy()
        
        # Update the DataFrame with these new columns
        for i, (m, s) in enumerate(zip(c_means, c_sds)):
            best_df.at[idx, f"Ch{i} Mean"] = m
            best_df.at[idx, f"Ch{i} SD"] = s

    # Cleanup: Drop the helper key and Sort
    best_df = best_df.drop(columns=['Model_Key', r'$p$'])# .sort_values('Model')

    cols_to_fmt = [c for c in best_df.columns if "Mean" in c or "SD" in c]
    best_df[cols_to_fmt] = best_df[cols_to_fmt].map(fmt)
    
    return best_df.set_index('Model').reindex(['FNO', 'CDeepONet', 'MDeepONet', 'CPodONet', 'MPodONet'])

'''
CISE plot
'''

def ms_df(data_dict, opt_out=['FNO']):
    # Initialize Criterion
    criterion = ChannelwiseMSE(dim_channel=1, dim_sample=0)
    all_results = []
    
    # Extract Ground Truth
    true = data_dict['y_tr']
    
    # Filter keys to get model predictions only
    model_keys = [k for k in data_dict.keys() if k not in ['x', 'y_tr']+opt_out]
    
    for model_name in model_keys:
        pred = data_dict[model_name]
        
        # Compute Metrics
        with torch.no_grad():
            c_means = criterion.cmean(true, pred).cpu().numpy()
            c_sds = criterion.csd(true, pred).cpu().numpy()
            g_mean = criterion.mean(true, pred).item()
            g_sd = criterion.sd(true, pred).item()

        # Initialize row
        res_row = {"Model_Full": model_name} # Keep original name for debugging
        
        # Add Global (Total) MSE
        res_row["Total Mean"] = g_mean
        res_row["Total SD"] = g_sd

        # Add Channel-wise results
        for i, (m, s) in enumerate(zip(c_means, c_sds)):
            res_row[f"Ch{i} Mean"] = m
            res_row[f"Ch{i} SD"] = s
                
        all_results.append(res_row)

    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # 1. Extract 'p' (digits inside parentheses). 
    # If no parentheses (e.g., 'FNO'), this becomes NaN.
    df['$p$'] = df['Model_Full'].str.extract(r'\((\d+)\)')
    
    # 2. Extract Model Name (Remove the parameter part if it exists)
    # This turns 'CPodONet(512)' -> 'CPodONet' and keeps 'FNO' -> 'FNO'
    df['Model'] = df['Model_Full'].str.replace(r'\(\d+\)', '', regex=True).str.strip()

    # Drop the temporary full name column
    df = df.drop(columns=['Model_Full'])

    return df

def plot_multiline_performance(df, var_names=None, use_log=False):
    # 1. Setup Data
    # Convert p to numeric; errors='coerce' turns NaN (from FNO) into NaNs
    df['$p$'] = pd.to_numeric(df['$p$'], errors='coerce')
    
    # Sort: Models first, then p. NaNs (FNO) usually end up at the end or beginning depending on pandas version
    df = df.sort_values(['Model', '$p$'])
    
    # Get unique p values for the X-axis (excluding NaN/FNO)
    unique_p = [256,512,1024,2048,4096] #sorted(df[df['$p$'].notna()]['$p$'].unique())
    x_indexes = np.arange(len(unique_p))
    
    # Identify channels
    channels = [c.replace(" Mean", "") for c in df.columns if "Ch" in c and "Mean" in c]
    num_channels = len(channels)
    
    # --- Grid Setup ---
    # Total on left, Channels on right (split into 2 rows)
    cols_per_row = (num_channels + 1) // 2 
    
    # Define widths: Total gets 1.5x width
    width_ratios = [1.5] + [1] * cols_per_row
    
    fig = plt.figure(figsize=(3.5 * (cols_per_row + 1), 5), dpi=200)
    gs = gridspec.GridSpec(2, cols_per_row + 1, width_ratios=width_ratios)
    
    # Create Subplots
    ax_total = fig.add_subplot(gs[:, 0]) # Total spans both rows
    
    chan_axes = []
    for r in range(2):
        for c in range(1, cols_per_row + 1):
            idx = r * cols_per_row + (c - 1)
            if idx < num_channels:
                ax = fig.add_subplot(gs[r, c])
                chan_axes.append(ax)

    all_axes = [ax_total] + chan_axes
    metrics = ["Total"] + channels
    display_names = ["Total"] + (var_names if var_names else channels)

    # 2. Plotting Logic
    # Added FNO to styles
    styles = {
        'CDeepONet': {'color': "#ff0eaf", 'marker': '^', 'linestyle': '-', 'label': 'cDeepONet'},
        'MDeepONet': {'color': "#ff870e", 'marker': '^', 'linestyle': '-', 'label': 'mDeepONet'},
        'CPodONet': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'cPOD-DeepONet'},
        'MPodONet': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-', 'label': 'mPOD-DeepONet'},
        'KPodONet': {'color': '#9467bd', 'marker': 's', 'linestyle': '-', 'label': 'mKPCA-DeepONet'},
        'FNO':      {'color': '#d62728', 'marker': None, 'linestyle': '--', 'label': 'FNO '}
    }


    for ax, metric, name in zip(all_axes, metrics, display_names):
        
        # Emphasize Total Plot
        if ax == ax_total:
            ax.set_facecolor('#fdfdfd')
            ax.set_title(name, fontsize=16, fontweight='bold')
        else:
            ax.set_title(name, fontsize=14)
        if use_log:
            ax.set_yscale('log')

        for model_name, style in styles.items():
            m_df = df[df['Model'] == model_name]
            if m_df.empty: continue
            
            # --- BASELINE LOGIC (FNO) ---
            if model_name == 'FNO':
                # FNO has no p dependence, so we take the mean of the available rows 
                # (usually just 1 row, but robust against duplicates)
                val_mean = m_df[f"{metric} Mean"].mean()
                val_sd = m_df[f"{metric} SD"].mean()
                
                # Plot horizontal line across the entire p range
                ax.plot(x_indexes, [val_mean]*len(x_indexes), 
                        linewidth=2.5, color=style['color'], linestyle=style['linestyle'])
                
                # Constant error band
                if not use_log or (val_mean - val_sd) > 0:
                    ax.fill_between(x_indexes, val_mean - val_sd, val_mean + val_sd, 
                                color=style['color'], alpha=0.1)

            # --- CURVE LOGIC (DeepONet, etc.) ---
            else:
                means = []
                sds = []
                for p in unique_p:
                    row = m_df[m_df['$p$'] == p]
                    if not row.empty:
                        means.append(row[f"{metric} Mean"].values[0])
                        sds.append(row[f"{metric} SD"].values[0])
                    else:
                        means.append(np.nan)
                        sds.append(np.nan)
                
                means = np.array(means)
                sds = np.array(sds)
                
                # Only plot if we have data
                if not np.all(np.isnan(means)):
                    ax.plot(x_indexes, means, linewidth=2.5, markersize=6, 
                            color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
                    ax.fill_between(x_indexes, means - sds, means + sds, 
                                    color=style['color'], alpha=0.15)

        # Formatting
        ax.set_xlabel(r'$p$', fontsize=12)
        ax.set_xticks(x_indexes)
        ax.set_xticklabels([str(p) for p in unique_p])
        ax.grid(True, linestyle=':', alpha=0.6)
        if not use_log:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        # Clean up labels for inner plots if needed
        # if ax != ax_total:
        #    ax.tick_params(labelleft=False)

    ax_total.set_ylabel('CISE' if not use_log else 'CISE(log)', fontsize=14)
    
    # 3. Legend
    # Filter legend to only show models present in the data
    present_models = [m for m in styles.keys() if m in df['Model'].unique()]
    
    handles = []
    for m in present_models:
        s = styles[m]
        # Create custom handle for legend based on style
        if s['marker'] is None: # FNO case
            h = plt.Line2D([0], [0], color=s['color'], linestyle=s['linestyle'], linewidth=2.5)
        else:
            h = plt.Line2D([0], [0], color=s['color'], marker=s['marker'], linestyle=s['linestyle'], linewidth=2.5)
        handles.append(h)

    fig.legend(handles=handles, 
               labels=[styles[m]['label'] for m in present_models], 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0), # Pushed slightly lower to avoid overlap
               ncol=len(present_models),     # Matches the number of labels exactly
               frameon=False, 
               fontsize=11,                  # Slightly smaller to fit width
               columnspacing=1.0,            # Adjusts space between columns
               handletextpad=0.5)            # Adjusts space between icon and text

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.show()

def plot_model_performance(df, scale=1, var_names=None, use_log=False):
    # --- CHANGE 1: Handle 'p' conversion safely ---
    # Use errors='coerce' so rows with no p (like FNO) become NaN instead of crashing
    df['$p$'] = pd.to_numeric(df['$p$'], errors='coerce') 
    
    df = df.sort_values(['Model', '$p$'])
    
    # Get unique p values ignoring NaNs (for the x-axis labels)
    unique_p = [256,512,1024,2048,4096] # sorted(df[df['$p$'].notna()]['$p$'].unique())
    x_indexes = np.arange(len(unique_p))
    
    # Identify channel counts
    channels = [c.replace(" Mean", "") for c in df.columns if "Ch" in c and "Mean" in c]
    metrics = ["Total"] + channels

    if var_names and len(var_names) == len(channels):
        display_names = ["Total"] + var_names
    else:
        display_names = ["Total"] + channels
    
    # Setup Figure
    width_ratios = [1.5] + [1.0] * (len(metrics) - 1)
    fig, axes = plt.subplots(1, len(metrics), 
                             figsize=(3.5 * sum(width_ratios), 3.5), 
                             dpi=200, 
                             gridspec_kw={'width_ratios': width_ratios})
    
    # --- CHANGE 2: Add FNO to your styles dictionary ---
    styles = {
        'CDeepONet': {'color': "#ff0eaf", 'marker': '^', 'linestyle': '-', 'label': 'cDeepONet'},
        'MDeepONet': {'color': "#ff870e", 'marker': '^', 'linestyle': '-', 'label': 'mDeepONet'},
        'CPodONet': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'cPOD-DeepONet'},
        'MPodONet': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-', 'label': 'mPOD-DeepONet'},
        'KPodONet': {'color': '#9467bd', 'marker': 's', 'linestyle': '-', 'label': 'mKPCA-DeepONet'},
        'FNO':      {'color': '#d62728', 'marker': None, 'linestyle': '--', 'label': 'FNO '}
    }

    if len(metrics) == 1: axes = [axes]
    

    for i, (ax, metric, name) in enumerate(zip(axes, metrics, display_names)):
        if i == 0:
            ax.set_facecolor('#fdfdfd') 
            ax.set_title(name, fontsize=17, fontweight='bold', color='#333333')
        else:
            ax.set_title(name, fontsize=15)
        
        if use_log:
            ax.set_yscale('log')

        for model in df['Model'].unique():
            m_df = df[df['Model'] == model]
            # Default fallback style if model not in dict
            style = styles.get(model, {'color': 'k', 'marker': 'd', 'linestyle': '-', 'label': model})
            
            # --- CHANGE 3: Branching Logic for FNO vs. Others ---
            if model == 'FNO':
                # Logic: Take the mean scalar value (ignoring p) and broadcast across x-axis
                val_mean = m_df[f"{metric} Mean"].mean() * scale
                val_sd = m_df[f"{metric} SD"].mean() * scale
                
                # Create a horizontal line spanning all p indices
                means = np.full(len(x_indexes), val_mean)
                lower = means - val_sd
                upper = means + val_sd
                
                # Plot with dashed line, no marker
                ax.plot(x_indexes, means, linewidth=2.5, 
                        color=style['color'], linestyle=style['linestyle'])
                # Fill band
                if not use_log or (val_mean - val_sd) > 0:
                    ax.fill_between(x_indexes, lower, upper, 
                                    color=style['color'], alpha=0.15)
                    
            else:
                # Original Logic for p-dependent models
                means = []
                sds = []
                for p in unique_p:
                    row = m_df[m_df['$p$'] == p]
                    if not row.empty:
                        means.append(row[f"{metric} Mean"].values[0] * scale)
                        sds.append(row[f"{metric} SD"].values[0] * scale)
                    else:
                        means.append(np.nan)
                        sds.append(np.nan)

                means = np.array(means)
                sds = np.array(sds)

                if np.all(np.isnan(means)): continue

                ax.plot(x_indexes, means, linewidth=2.5, markersize=6, 
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
                
                ax.fill_between(x_indexes, means - sds, means + sds, 
                                color=style['color'], alpha=0.15)

        # Formatting
        ax.set_xlabel(r'$p$', fontsize=16)
        ax.set_xticks(x_indexes)
        ax.set_xticklabels([str(int(p)) for p in unique_p]) # Ensure p labels are ints
        if not use_log:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if ax == axes[0]:
            ax.set_ylabel('CISE' if not use_log else 'CISE(log)', fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    
    # Legend construction
    present_models = [m for m in styles.keys() if m in df['Model'].unique()]
    handles = []
    for m in present_models:
        s = styles[m]
        # Handle legend entry for FNO (no marker) vs others
        if s['marker'] is None:
             h = plt.Line2D([0], [0], color=s['color'], linestyle=s['linestyle'], linewidth=2.5)
        else:
             h = plt.Line2D([0], [0], color=s['color'], marker=s['marker'], linestyle=s['linestyle'])
        handles.append(h)

    fig.legend(handles=handles, 
               labels=[styles[m]['label'] for m in present_models], 
               loc='lower center', 
               bbox_to_anchor=(0.5, -0.08), # Pushed slightly lower to avoid overlap
               ncol=len(present_models),     # Matches the number of labels exactly
               frameon=False, 
               fontsize=11,                  # Slightly smaller to fit width
               columnspacing=1.0,            # Adjusts space between columns
               handletextpad=0.5)            # Adjusts space between icon and text
    plt.show()

'''
Pred & Error Plot
'''

# Global settings for clean axes
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

def plot_TE(data_dict, model_key=None, mode='both', sample_idx=0,
            labels=[r'Re($E$)',r'Im($E$)', r'$T$'], 
            cmaps=['RdBu_r','RdBu_r', 'magma']):

    input_key, target_key = 'x', 'y_tr'
    model_keys = model_key if model_key is not None else [k for k in data_dict.keys() if k not in ['x', 'y_tr']]
    N, C, W, H = data_dict[model_keys[0]].shape
    
    rows_per_phy = 2 if mode == 'both' else 1
    num_rows = C * rows_per_phy
    num_cols = 2 + len(model_keys) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5* num_cols, 3.2* num_rows), squeeze=False)

    tru = data_dict[target_key][sample_idx].detach().cpu()
    inp = data_dict[input_key][sample_idx].detach().cpu()

    for p_idx in range(C):
        # --- RAW ROWS ---
        r = p_idx
        # Slot 0: Input x (only in Top-Left)
        if p_idx == 0:
            axes[r, 0].imshow(inp[0], cmap='viridis')
            axes[r, 0].set_title(r"$\sigma/\kappa$", fontsize=20)
        else:
            axes[r, 0].axis('off')
        
        ax_target = axes[r, 1]
        im_target = ax_target.imshow(tru[p_idx], cmap=cmaps[p_idx])
        if r == 0: ax_target.set_title("Target", fontsize=20)
        ax_target.set_ylabel(labels[p_idx], fontsize=20)
            
        if mode in ['raw', 'both']:
            # Prepare data: Target + Predictions
            model_vals = [data_dict[k][sample_idx][p_idx].detach().cpu() for k in model_keys]
            v_min, v_max = min(m.min() for m in model_vals), max(m.max() for m in model_vals)

            im_target.set_clim(v_min, v_max)

            for c_idx, img_data in enumerate(model_vals):
                ax = axes[r, c_idx + 2]
                im = ax.imshow(img_data, vmin=v_min, vmax=v_max, cmap=cmaps[p_idx])
                
                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if model_name == 'KPodONet':
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        # e.g., 'mPodONet' -> 'mPOD-DeepONet'
                        model_name = model_name[0].lower() + 'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)

                # FIX: Attach colorbar to the last column of the row
                if c_idx == len(model_vals) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)

        # --- ERROR ROWS ---
        if mode in ['error', 'both']:
            r = p_idx + C if mode == 'both' else p_idx
            start_c = 2 # if mode == 'both' else 0
            
            if mode == 'both':
                axes[r, 0].axis('off')
                axes[r, 1].axis('off')

            errors = [data_dict[k][sample_idx][p_idx].detach().cpu() - tru[p_idx] for k in model_keys]
            max_error = max(e.abs().max() for e in errors)
            v_min, v_max = -max_error, max_error

            for c_idx, err_data in enumerate(errors):
                ax = axes[r, c_idx + start_c]
                im = ax.imshow(err_data, vmin=v_min, vmax=v_max, cmap='bwr')

                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if model_name == 'KPodONet':
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        # e.g., 'mPodONet' -> 'mPOD-DeepONet'
                        model_name = model_name[0].lower() + 'POD-DeepONet'
                    elif 'DeepONet' in model_name: 
                        # e.g., 'DeepONet' -> 'deepONet'
                        model_name = model_name[0].lower() + model_name[1:]
                
                if c_idx == 0:
                    ax.set_ylabel(f"{labels[p_idx]} Error" if mode =='error' else f"{labels[p_idx]}\nError", fontsize=20)

                # FIX: Attach colorbar to the last error column
                if c_idx == len(errors) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(
        left=0.1,   # Room for Y-labels
        right=0.90,  # Room for Colorbars
        wspace=0.05,  # Horizontal gap between models
        hspace=0.1   # Vertical gap between rows
    )
    # plt.tight_layout()
    plt.show()


def plot_NS(data_dict, model_key=None, mode='both', sample_idx=0,
            labels=[r'$|u|$', r'$T$'], 
            cmaps=['RdPu', 'magma']):

    input_key, target_key = 'x', 'y_tr'
    model_keys = model_key if model_key is not None else [k for k in data_dict.keys() if k not in ['x', 'y_tr']]
    N, C, W, H = data_dict[model_keys[0]].shape
    
    num_phys_vars = 2
    rows_per_phy = 2 if mode == 'both' else 1
    num_rows = num_phys_vars * rows_per_phy
    num_cols = 2 + len(model_keys) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5* num_cols, 3.2* num_rows), squeeze=False)

    def get_phys_val(data, p_type):
        if p_type == 0: # Norm of first two channels
            return torch.linalg.norm(data[:2], dim=0).detach().cpu()
        else: # Remaining channel (C2)
            return data[2].detach().cpu()

    tru = data_dict[target_key][sample_idx]
    inp = data_dict[input_key][sample_idx].detach().cpu()

    for p_idx in range(num_phys_vars):
        # --- RAW ROWS ---
        
        r = p_idx            
        # Slot 0: Input x (only in Top-Left)
        if p_idx == 0:
            axes[r, 0].imshow(inp[0], cmap='viridis')
            axes[r, 0].set_title(r"$Q$", fontsize=20)
        else:
            axes[r, 0].axis('off')

            # Prepare data: Target + Predictions
        target_val = get_phys_val(tru, p_idx)
        ax_target = axes[r, 1]
        im_target = ax_target.imshow(target_val, cmap=cmaps[p_idx])
        if r == 0: ax_target.set_title("Target", fontsize=20)
        ax_target.set_ylabel(labels[p_idx], fontsize=20)
            

        if mode in ['raw', 'both']:
            model_vals = [get_phys_val(data_dict[k][sample_idx], p_idx) for k in model_keys]
            v_min, v_max = min(m.min() for m in model_vals), max(m.max() for m in model_vals)
            im_target.set_clim(v_min, v_max)

            for c_idx, img_data in enumerate(model_vals):
                ax = axes[r, c_idx + 2]
                im = ax.imshow(img_data, vmin=v_min, vmax=v_max, cmap=cmaps[p_idx])
                
                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if 'KPodONet' in model_name:
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        model_name = model_name[0].lower()+'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)

                # FIX: Attach colorbar to the last column of the row
                if c_idx == len(model_vals) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)

        # --- ERROR ROWS ---
        if mode in ['error', 'both']:
            r = p_idx + num_phys_vars if mode == 'both' else p_idx
            start_c = 2 # if mode == 'both' else 0
            
            if mode == 'both':
                axes[r, 0].axis('off')
                axes[r, 1].axis('off')

            target_val = get_phys_val(tru, p_idx)
            errors = [get_phys_val(data_dict[k][sample_idx], p_idx) - target_val for k in model_keys]
            max_error = max(e.abs().max() for e in errors)
            v_min, v_max = -max_error, max_error

            for c_idx, err_data in enumerate(errors):
                ax = axes[r, c_idx + start_c]
                im = ax.imshow(err_data, vmin=v_min, vmax=v_max, cmap='bwr')

                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if model_name == 'KPodONet':
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        # e.g., 'mPodONet' -> 'mPOD-DeepONet'
                        model_name = model_name[0].lower() + 'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)
                
                if c_idx == 0:
                    ax.set_ylabel(f"{labels[p_idx]} Error" if mode =='error' else f"{labels[p_idx]}\nError", fontsize=20)

                # FIX: Attach colorbar to the last error column
                if c_idx == len(errors) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(
        left=0.1,   # Room for Y-labels
        right=0.90,  # Room for Colorbars
        wspace=0.05,  # Horizontal gap between models
        hspace=0.1   # Vertical gap between rows
    )
    plt.show()

def plot_Eflow(data_dict, model_key=None, mode='both', sample_idx=0,
            labels= [r'$|u|$', r'$V$'], 
            cmaps=['RdPu', 'Spectral']):

    input_key, target_key = 'x', 'y_tr'
    model_keys = model_key if model_key is not None else [k for k in data_dict.keys() if k not in ['x', 'y_tr']]
    N, C, W, H = data_dict[model_keys[0]].shape
    
    num_phys_vars = 2
    rows_per_phy = 2 if mode == 'both' else 1
    num_rows = num_phys_vars * rows_per_phy
    num_cols = 2 + len(model_keys) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5* num_cols, 3.2* num_rows), squeeze=False)

    def get_phys_val(data, p_type):
        if p_type == 0: # Norm of first two channels
            return torch.linalg.norm(data[1:3], dim=0).detach().cpu()
        else: # Remaining channel (C2)
            return data[0].detach().cpu()

    tru = data_dict[target_key][sample_idx]
    inp = data_dict[input_key][sample_idx].detach().cpu()

    for p_idx in range(num_phys_vars):
        # --- RAW ROWS ---
        
        r = p_idx            
        # Slot 0: Input x (only in Top-Left)
        if p_idx == 0:
            axes[r, 0].imshow(inp[0], cmap='viridis')
            axes[r, 0].set_title(r"$\sigma$", fontsize=20)
        else:
            axes[r, 0].axis('off')

        # Prepare data: Target + Predictions

        target_val = get_phys_val(tru, p_idx)
        ax_target = axes[r, 1]
        im_target = ax_target.imshow(target_val, cmap=cmaps[p_idx])
        if r == 0: ax_target.set_title("Target", fontsize=20)
        ax_target.set_ylabel(labels[p_idx], fontsize=20)

        if mode in ['raw', 'both']:
            model_vals = [get_phys_val(data_dict[k][sample_idx], p_idx) for k in model_keys]
            v_min, v_max = min(m.min() for m in model_vals), max(m.max() for m in model_vals)
            im_target.set_clim(v_min, v_max)

            for c_idx, img_data in enumerate(model_vals):
                ax = axes[r, c_idx + 2]
                im = ax.imshow(img_data, vmin=v_min, vmax=v_max, cmap=cmaps[p_idx])
                
                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if 'KPodONet' in model_name:
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        model_name = model_name[0].lower()+'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)

                # FIX: Attach colorbar to the last column of the row
                if c_idx == len(model_vals) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)

        # --- ERROR ROWS ---
        if mode in ['error', 'both']:
            r = p_idx + num_phys_vars if mode == 'both' else p_idx
            start_c = 2 # if mode == 'both' else 0
            
            if mode == 'both':
                axes[r, 0].axis('off')
                axes[r, 1].axis('off')

            target_val = get_phys_val(tru, p_idx)
            errors = [get_phys_val(data_dict[k][sample_idx], p_idx) - target_val for k in model_keys]
            max_error = max(e.abs().max() for e in errors)
            v_min, v_max = -max_error, max_error

            for c_idx, err_data in enumerate(errors):
                ax = axes[r, c_idx + start_c]
                im = ax.imshow(err_data, vmin=v_min, vmax=v_max, cmap='bwr')

                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if model_name == 'KPodONet':
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        # e.g., 'mPodONet' -> 'mPOD-DeepONet'
                        model_name = model_name[0].lower() + 'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)
                
                if c_idx == 0:
                    ax.set_ylabel(f"{labels[p_idx]} Error" if mode =='error' else f"{labels[p_idx]}\nError", fontsize=20)

                # FIX: Attach colorbar to the last error column
                if c_idx == len(errors) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)

    plt.subplots_adjust(
        left=0.1,   # Room for Y-labels
        right=0.90,  # Room for Colorbars
        wspace=0.05,  # Horizontal gap between models
        hspace=0.1   # Vertical gap between rows
    )
    plt.show()

def plot_MHD(data_dict, model_key=None, mode='both', sample_idx=0,
            labels= [r'$|J|$', r'$|u|$'], 
            cmaps=['RdPu', 'YlGnBu']):

    input_key, target_key = 'x', 'y_tr'
    model_keys = model_key if model_key is not None else [k for k in data_dict.keys() if k not in ['x', 'y_tr']]
    N, C, W, H = data_dict[model_keys[0]].shape
    
    num_phys_vars = 2
    rows_per_phy = 2 if mode == 'both' else 1
    num_rows = num_phys_vars * rows_per_phy
    num_cols = 2 + len(model_keys) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5* num_cols, 3.2* num_rows), squeeze=False)

    def get_phys_val(data, p_type):
        if p_type == 0: # Norm of first two channels
            return torch.linalg.norm(data[0:3], dim=0).detach().cpu()
        else: # Remaining channel (C2)
            return  torch.linalg.norm(data[3:5], dim=0).detach().cpu()

    tru = data_dict[target_key][sample_idx]
    inp = data_dict[input_key][sample_idx].detach().cpu()

    for p_idx in range(num_phys_vars):
        # --- RAW ROWS ---
        
        r = p_idx            
        # Slot 0: Input x (only in Top-Left)
        if p_idx == 0:
            axes[r, 0].imshow(inp[0], cmap='viridis')
            axes[r, 0].set_title(r"$B_z$", fontsize=20)
        else:
            axes[r, 0].axis('off')
        
        target_val = get_phys_val(tru, p_idx)
        ax_target = axes[r, 1]

        model_vals = [get_phys_val(data_dict[k][sample_idx], p_idx) for k in model_keys]
        v_min, v_max = min(m.min() for m in model_vals), max(m.max() for m in model_vals)
        
        if p_idx ==0:
            im_target = ax_target.imshow(target_val, norm=LogNorm(vmin=max(1e-3, v_min), vmax=v_max),cmap=cmaps[p_idx])
            ax_target.set_ylabel(f"{labels[p_idx]}(Log)", fontsize=20)
        else:
            im_target = ax_target.imshow(target_val, cmap=cmaps[p_idx])
            
            ax_target.set_ylabel(labels[p_idx], fontsize=20)
        if r == 0: ax_target.set_title("Target", fontsize=20)

        if mode in ['raw', 'both']:
            im_target.set_clim(v_min, v_max)

            for c_idx, img_data in enumerate(model_vals):
                ax = axes[r, c_idx + 2]
                if p_idx ==0:
                    im = ax.imshow(img_data, norm=LogNorm(vmin=max(1e-3, v_min), vmax=v_max),cmap=cmaps[p_idx])
                else:
                    im = ax.imshow(img_data, vmin=v_min, vmax=v_max, cmap=cmaps[p_idx])
                
                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if 'KPodONet' in model_name:
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        model_name = model_name[0].lower()+'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)
                

                # FIX: Attach colorbar to the last column of the row
                if c_idx == len(model_vals) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)

        # --- ERROR ROWS ---
        if mode in ['error', 'both']:
            r = p_idx + num_phys_vars if mode == 'both' else p_idx
            start_c = 2 # if mode == 'both' else 0
            
            if mode == 'both':
                axes[r, 0].axis('off')
                axes[r, 1].axis('off')

            target_val = get_phys_val(tru, p_idx)
            errors = [get_phys_val(data_dict[k][sample_idx], p_idx) - target_val for k in model_keys]
            max_error = max(e.abs().max() for e in errors)
            v_min, v_max = -max_error, max_error

            for c_idx, err_data in enumerate(errors):
                ax = axes[r, c_idx + start_c]
                if p_idx ==0:
                    im = ax.imshow(err_data, norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=v_min, vmax=v_max, base=10), cmap='bwr')
                    if c_idx == 0:
                        ax.set_ylabel(f"{labels[p_idx]} Error(SymLog)" if mode =='error' else f"{labels[p_idx]}\nError(SymLog)", fontsize=20)
                
                else:
                    im = ax.imshow(err_data, vmin=v_min, vmax=v_max, cmap='bwr')
                    if c_idx == 0:
                        ax.set_ylabel(f"{labels[p_idx]} Error" if mode =='error' else f"{labels[p_idx]}\nError", fontsize=20)
                        
                if r == 0:
                    model_name = re.search(r'([a-zA-Z]+)',model_keys[c_idx]).group(1)
                    if model_name == 'KPodONet':
                        model_name = 'mKPCA-DeepONet'
                    elif 'PodONet' in model_name:
                        # e.g., 'mPodONet' -> 'mPOD-DeepONet'
                        model_name = model_name[0].lower() + 'POD-DeepONet'
                    elif 'DeepONet'in model_name: 
                        model_name = model_name[0].lower() + model_name[1:]
                    ax.set_title(model_name, fontsize=20)

                # FIX: Attach colorbar to the last error column
                if c_idx == len(errors) - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad=0.1)
                    fig.colorbar(im, cax=cax)
    plt.subplots_adjust(
        left=0.1,   # Room for Y-labels
        right=0.90,  # Room for Colorbars
        wspace=0.05,  # Horizontal gap between models
        hspace=0.1   # Vertical gap between rows
    )
    plt.show()