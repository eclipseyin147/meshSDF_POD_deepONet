import gc
import torch
import math
import torch.nn as nn

class MFPCA_RF_EKPCA(nn.Module):
    def __init__(self, M_univariate, M_multivariate, 
                 n_random_features=100, 
                 n_channels=1, spatial_dims=(64, 64),
                 kernel='rbf', gamma=None, ):
        super().__init__()
        self.M_u_val = M_univariate # Store for reference
        self.M_m = M_multivariate
        self.m = n_random_features
        self.kernel = kernel
        self.gamma = gamma
        self.n_channels = n_channels
        W, H = spatial_dims

        # 1. Global Metadata
        self.register_buffer('is_fitted', torch.tensor(False))
        self.register_buffer('input_shape', torch.tensor([n_channels, W, H], dtype=torch.long))

        # 2. Pre-allocate Global PCA Buffers
        # Univariate scores total dimension
        total_u_dim = n_channels * M_univariate if isinstance(M_univariate, int) else sum(M_univariate)
        
        # Random Feature Weights: (Total Univariate Dim, m/2)
        self.register_buffer('rf_weights', torch.zeros((total_u_dim, n_random_features // 2)))
        
        # Global PCA Components: (m, M_multivariate)
        self.register_buffer('pca_components', torch.zeros((n_random_features, M_multivariate)))
        self.register_buffer('pca_singular_values', torch.zeros(M_multivariate))
        
        # Pre-image Mapper: (m, Total Univariate Dim)
        self.register_buffer('pre_image_mapper', torch.zeros((n_random_features, total_u_dim)))

        # 3. Pre-allocate Channel-specific SVD Bases
        # Flattened spatial dim is W*H
        spatial_dim = W * H
        if isinstance(M_univariate, int):
            self.M_u_list = [M_univariate] * n_channels
        else:
            self.M_u_list = M_univariate

        for c in range(n_channels):
            # Basis V shape: (W*H, M_u)
            self.register_buffer(f'basis_c{c}', torch.zeros((spatial_dim, self.M_u_list[c])))
            self.register_buffer(f'singular_values_c{c}', torch.zeros(self.M_u_list[c]))

    def _compute_features(self, X):
        if self.kernel == 'linear':
            return X
        # Projection: (N, total_u_dim) @ (total_u_dim, m/2) -> (N, m/2)
        projection = torch.matmul(X, self.rf_weights)
        Z = torch.cat([torch.cos(projection), torch.sin(projection)], dim=1)
        return Z * math.sqrt(2.0 / self.m)

    def fit(self, X_tensor):
        """
        Fits the model. Since shapes are already allocated, we use .copy_() 
        to update buffers in-place.
        """
        N, C, W, H = X_tensor.shape
        univariate_scores_list = []

        # --- STEP 1: Univariate FPCA ---
        for c in range(C):
            X_c = X_tensor[:, c, :, :].reshape(N, -1)
            q = self.M_u_list[c]
            
            # Note: pca_lowrank returns V of shape (Features, q)
            _, S, V = torch.pca_lowrank(X_c, q=q, center=False, niter=3)
            
            # Update buffers in-place
            getattr(self, f'basis_c{c}').copy_(V)
            getattr(self, f'singular_values_c{c}').copy_(S)
            
            univariate_scores_list.append(torch.matmul(X_c, V))

        train_Xi = torch.cat(univariate_scores_list, dim=1)
        
        # --- STEP 2: Random Features ---
        if self.kernel == 'rbf':
            if self.gamma is None: self.gamma = 1.0 / train_Xi.shape[1]
            std = math.sqrt(2 * self.gamma)
            # Generate weights and copy to buffer
            new_weights = torch.randn_like(self.rf_weights) * std
            self.rf_weights.copy_(new_weights)

        # --- STEP 3: Global PCA ---
        Z = self._compute_features(train_Xi)
        _, S_z, V_z = torch.svd_lowrank(Z, q=self.M_m, niter=3)
        self.pca_components.copy_(V_z)
        self.pca_singular_values.copy_(S_z)

        # --- STEP 4: Pre-image (Ridge Regression) ---
        ridge_alpha = 1e-4
        Z_t_Z = torch.matmul(Z.T, Z)
        reg = ridge_alpha * torch.eye(Z_t_Z.shape[0], device=Z.device)
        Z_t_Xi = torch.matmul(Z.T, train_Xi)
        mapper = torch.linalg.solve(Z_t_Z + reg, Z_t_Xi)
        self.pre_image_mapper.copy_(mapper)
        
        # self.is_fitted.fill_(True)
        return self

    def transform(self, X_tensor):
        # if not self.is_fitted:
        #     raise RuntimeError("Model must be fitted or loaded before transform.")
        
        N, C, W, H = X_tensor.shape
        u_scores = []
        for c in range(C):
            X_c = X_tensor[:, c, :, :].reshape(N, -1)
            V = getattr(self, f'basis_c{c}')
            u_scores.append(torch.matmul(X_c, V))

        Xi_new = torch.cat(u_scores, dim=1)
        Z_new = self._compute_features(Xi_new)
        return torch.matmul(Z_new, self.pca_components)

    def reconstruct(self, rho):
        # if not self.is_fitted:
        #     raise RuntimeError("Model must be fitted or loaded before reconstruction.")
            
        N = rho.shape[0]
        C, W, H = self.input_shape.tolist()

        # Map back from PCA space to Univariate Score space
        Z_rec = torch.matmul(rho, self.pca_components.T)
        Xi_rec = torch.matmul(Z_rec, self.pre_image_mapper)

        X_rec_list = []
        curr_idx = 0
        for c in range(C):
            M_c = self.M_u_list[c]
            V = getattr(self, f'basis_c{c}')
            xi_c = Xi_rec[:, curr_idx : curr_idx + M_c]
            curr_idx += M_c
            
            # X_approx = Scores * V^T
            flat_rec = torch.matmul(xi_c, V.T)
            X_rec_list.append(flat_rec.reshape(N, 1, W, H))

        return torch.cat(X_rec_list, dim=1)

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

class MKPodONet(nn.Module):
    def __init__(self, p=256, gamma=1e-6, 
                 kernel='rbf', # 'linear', 'rbf'
                 in_channels=1, out_channels=3, 
                 grid_size=[128,128],
        ):
        super().__init__()

        self.mfpca_y = MFPCA_RF_EKPCA(
            M_univariate= int(grid_size[0]*grid_size[1]/2), #
            M_multivariate=p,
            n_random_features=p,
            n_channels=out_channels, 
            spatial_dims=grid_size, 
            kernel=kernel,
            gamma=gamma)
        
        self.branch_net = BranchNetConv(
                p=p, in_channels=in_channels)
            
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


