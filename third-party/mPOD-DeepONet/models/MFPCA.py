import gc
import torch
import math
import torch.nn as nn

class MFPCA_Modular(nn.Module):
    def __init__(self, n_components: list[int] | int,
                 n_channels=int, spatial_dims=list[int], 
                 joint_pca: bool = True, 
                 reduce: bool = True,
                 joint_k: int = None):
        super().__init__()
        self.n_components = n_components
        self.joint_pca = joint_pca  # Toggle between Multivariate and Channel-wise
        self.reduce = reduce
        self.C = n_channels

        # Model attributes
        self.D_flat = math.prod(spatial_dims)
        self.spatial_rank = len(spatial_dims)
        self.register_buffer('univariate_components_buffer', 
                             torch.zeros((n_channels, n_components, self.D_flat)))
        total_k = n_channels * n_components
        if self.joint_pca:
            if self.reduce:
                self.joint_k = joint_k if joint_k is not None else int(total_k/2)
                self.register_buffer('joint_eigenvectors_buffer', 
                                    torch.zeros((self.joint_k, total_k)))
            else:
                self.joint_k = total_k
                self.register_buffer('joint_eigenvectors_buffer', 
                                    torch.zeros((total_k, total_k)))
        else:
            self.register_buffer('joint_eigenvectors_buffer', 
                                 torch.empty(0))
        
        self.register_buffer('spatial_shape_buffer', 
                             torch.zeros(self.spatial_rank, dtype=torch.long))
        self.Ks = [n_components] * self.C

    def _pca(self, X: torch.Tensor, k: int):
        """
        Performs Randomized PCA using torch.svd_lowrank (Uncentered).
        X: (N, D) input tensor
        k: target rank (number of components)
        """
        # Optimized Randomized SVD
        U, S, V = torch.svd_lowrank(X, q=k, niter=2)
        scores = U @ torch.diag(S)
        return scores, V.T

    def fit(self, X: torch.Tensor):
        """
        Fits the MFPCA model to the data X.

        Args:
            X: Tensor of shape (N, C, ...)
        """
        N, C = X.shape[0], X.shape[1]
        spatial_shape = X.shape[2:]

        # 1. Determine components per channel
        if isinstance(self.n_components, int):
            self.Ks = [self.n_components] * C
        else:
            self.Ks = self.n_components

        X_flat = X.reshape(N, C, -1)
        all_univariate_scores = []
        all_phi = []

        # --- Step 1: Univariate PCA (Always performed) ---
        for j in range(C):
            
            scores_j, phi_j = self._pca(X_flat[:, j, :], k=self.Ks[j])
            all_univariate_scores.append(scores_j)
            all_phi.append(phi_j)

            del phi_j
            torch.cuda.empty_cache()

        self.register_buffer('univariate_components_buffer', torch.stack(all_phi))
        self.register_buffer('spatial_shape_buffer', torch.tensor(spatial_shape, dtype=torch.long)) 

        # --- Step 2: Joint PCA (Conditional) ---
        if self.joint_pca:
            Xi = torch.cat(all_univariate_scores, dim=1)
            total_comps = sum(self.Ks)
            n_joint = min(N, total_comps, self.joint_k)
            
            # Learn the coupling matrix
            joint_scores, joint_components = self._pca(Xi, k=n_joint)
            self.multivariate_scores = joint_scores.cpu()
            # self.joint_eigenvectors = joint_components
            # self.joint_eigenvalues = torch.var(joint_scores, dim=0, unbiased=True)
            self.register_buffer('joint_eigenvectors_buffer', joint_components)

            del joint_scores, Xi, all_univariate_scores
            torch.cuda.empty_cache()
            
            ## Use Branch Net
            # # --- Step 4: Multivariate Eigenfunctions ---
            # # [cite_start]Eq (9): psi_m^(j) = sum [c_m]_n^(j) * phi_n^(j) [cite: 11] 

        else:
            # In Channel-wise mode, the "joint" matrix is just an Identity matrix
            # This means no mixing between channel scores occurs
            self.joint_eigenvectors_buffer = torch.empty(0)

    def transform(self, X: torch.Tensor):
        N, C = X.shape[0], X.shape[1]
        X_flat = X.reshape(N, C, -1)

        # 1. Get univariate scores
        univariate_scores = []
        for j in range(C):
            phi_j = self.univariate_components_buffer[j]
            univariate_scores.append(X_flat[:, j, :] @ phi_j.T)
        
        Xi = torch.cat(univariate_scores, dim=1)

        # 2. Apply joint coupling if enabled
        if self.joint_pca and self.joint_eigenvectors_buffer.numel() > 0:
            return Xi @ self.joint_eigenvectors_buffer.T
        return Xi

    def reconstruct(self, scores: torch.Tensor):
        N = scores.shape[0]
        C = self.univariate_components_buffer.shape[0]

        # 1. Undo Joint PCA using buffer
        if self.joint_pca and self.joint_eigenvectors_buffer.numel() > 0:
            Xi_recon = scores @ self.joint_eigenvectors_buffer
        else:
            Xi_recon = scores

        # 2. Undo Univariate PCA
        recon_channels = []
        current_idx = 0
        for j in range(C):
            k_j = self.Ks[j]
            scores_j = Xi_recon[:, current_idx : current_idx + k_j]
            
            # Use j-th slice of univariate buffer
            X_j_recon = scores_j @ self.univariate_components_buffer[j]
            recon_channels.append(X_j_recon)
            current_idx += k_j

        # 3. Reshape
        X_recon = torch.stack(recon_channels, dim=1)
        spatial_dim = tuple(self.spatial_shape_buffer.tolist())
        return X_recon.view((N, C) + spatial_dim)


class BranchNetLinear(nn.Module):
    """處理輸入函數（初始溫度場T0）的PCA Net，使用MLP結構"""
    def __init__(self, p_in, p_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(p_in, 2*p_in),
            nn.ReLU(),
            nn.Linear(2*p_in, 2*p_in),
            nn.ReLU(),
            nn.Linear(2*p_in, p_out)
        )

    def forward(self, x):
        # T0: [batch, 1, p_in]
        return self.fc(x)  # [batch, p_out]

class BranchNetConv(nn.Module):
    """from Multiphysics Bench DeepONet"""
    def __init__(self, p=256, in_channels = 1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # [128,128] -> [128,128]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [64,64]
            nn.Conv2d(32, 128, kernel_size=3, padding=1),  # [128,128] -> [128,128]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [64,64]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> [64,64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32,32]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [32,32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> [4,4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, p) 
        )

    def forward(self, T0):
        # T0: [batch, 1, 128, 128]
        x = self.conv_layers(T0)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [batch, p]

class MPodONet(nn.Module):
    def __init__(self, p=256, in_channels=1, out_channels=3, 
                 grid_size=[128,128], branch_net = 'Conv'):
        super().__init__()
        self.uni_p = int(grid_size[0]*grid_size[1]/2)
        self.joint_p = p

        self.mfpca_y = MFPCA_Modular(
            n_components=self.uni_p,
            n_channels=out_channels, 
            spatial_dims=grid_size, 
            joint_pca=True,
            reduce=True,
            joint_k=self.joint_p)
        
        if branch_net == 'Linear':
            hidden = 2*self.joint_p
            self.branch_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * grid_size[0] * grid_size[1], hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.joint_p)
            )
        else:
            self.branch_net = BranchNetConv(
                in_channels=in_channels, 
                p=self.joint_p)

    def pca_fit(self, y_train, device='cpu'):
        y_train = y_train.to(device)
        self.mfpca_y.fit(y_train)
        del y_train
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x):
        scores = self.branch_net(x)
        y_pred = self.mfpca_y.reconstruct(scores)
        return y_pred

class ChannelPodONet(nn.Module):
    def __init__(self, p=256, in_channels=1, out_channels=3, 
                 grid_size=[128,128], branch_net = 'Conv'):
        '''
        p: PCA n_components
        grid_size: Gird size for MFPCA init
        branch_net: 'Linear', 'Pointwise', 'Conv', 'Hybrid'
        '''
        super().__init__()
        # Use the Modular version with joint_pca=False for independent channels
        self.mfpca_y = MFPCA_Modular(
            n_components=p,
            n_channels=out_channels, 
            spatial_dims=grid_size, 
            joint_pca=False)
        
        # The Branch Net must output the total sum of components
        # If p=256 and channels=3, output is 768

        self.total_p = p * out_channels
        if branch_net == 'Linear':
            hidden = 2*self.total_p
            self.branch_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * grid_size[0] * grid_size[1], hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.total_p)
            )
        elif branch_net == 'Conv':
            self.branch_net = BranchNetConv(
                p=self.total_p, in_channels=in_channels)


    def pca_fit(self, y_train, device='cpu'):
        y_train = y_train.to(device)
        self.mfpca_y.fit(y_train)
        del y_train
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, x):
        # 1. Branch Net predicts the "Channel-wise concatenated scores"
        # Shape: (Batch, total_p)
        scores = self.branch_net(x) 
        
        # 2. Reconstruct using the independent channel bases
        # Shape: (Batch, C, H, W)
        y_pred = self.mfpca_y.reconstruct(scores)
        return y_pred

    
