import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecisionStructure:
    """
    Helper class to manage the structure of the decision vector.
    v = [binary_part, permutation_part, continuous_part]
    """
    def __init__(self, binary_dim=0, permutation_dims=None, continuous_dim=0):
        """
        Args:
            binary_dim (int): Number of binary variables.
            permutation_dims (list of tuples): List of (n, k) for each permutation set.
                                             Encoded as n*k one-hot vector.
            continuous_dim (int): Number of continuous variables.
        """
        self.binary_dim = binary_dim
        self.permutation_dims = permutation_dims if permutation_dims else []
        self.continuous_dim = continuous_dim
        
        self.perm_starts = []
        self.perm_ends = []
        self.total_perm_dim = 0
        
        current_idx = binary_dim
        for n, k in self.permutation_dims:
            size = n * k
            self.perm_starts.append(current_idx)
            self.perm_ends.append(current_idx + size)
            self.total_perm_dim += size
            current_idx += size
            
        self.total_dim = binary_dim + self.total_perm_dim + continuous_dim

    def decode_raw_output(self, raw_output):
        """
        Applies typed activation functions to raw network output.
        Args:
            raw_output (Tensor): [batch, total_dim]
        Returns:
            Tensor: Post-processed output.
        """
        batch_size = raw_output.shape[0]
        parts = []
        
        # Binary part
        if self.binary_dim > 0:
            bin_part = raw_output[:, :self.binary_dim]
            parts.append(torch.sigmoid(bin_part))
            
        # Permutation parts
        if self.permutation_dims:
            for i, (n, k) in enumerate(self.permutation_dims):
                start = self.perm_starts[i]
                end = self.perm_ends[i]
                perm_part = raw_output[:, start:end]
                # Reshape to (batch, n, k) to apply softmax over k (assignments)? 
                # Or softmax over n (positions)? 
                # Spec says: "Softmax per position". Assuming (n, k) one-hot flattened.
                # Usually for ordered set of size k from n items.
                # Let's reshape to (batch, k, n) or (batch, n, k). 
                # "ordered set (size k): n * k one-hot" -> implies k slots, n choices each.
                # We want to select one item per slot.
                # Apply softmax over the 'n' dimension for each of the 'k' slots.
                # Wait, if it is n*k flattened, is it k vectors of size n?
                # Spec: "(n x k) one-hot (flattened)"
                # "Permutation blocks: Softmax per position"
                # Let's assume the encoding is: [slot 1 (size n), slot 2 (size n), ..., slot k (size n)]
                # So we reshape to (batch, k, n) and apply softmax dim=2.
                
                perm_part_reshaped = perm_part.view(batch_size, k, n)
                perm_soft = F.softmax(perm_part_reshaped, dim=2)
                parts.append(perm_soft.view(batch_size, -1))
        
        # Continuous part
        if self.continuous_dim > 0:
            cont_start = self.binary_dim + self.total_perm_dim
            cont_part = raw_output[:, cont_start:]
            parts.append(cont_part)
            
        return torch.cat(parts, dim=1)

    def encode_solution(self, solutions, device='cpu'):
        """
        Encodes a list of Solution objects into a single tensor v.
        """
        # We assume solutions have .int_values and .dbl_values matching the structure.
        # Structure definition:
        # binary_dim corresponds to one or more BOOL sets?
        # permutation_dims correspond to OS/UOS/OMS/UOMS sets?
        # This mapping is a bit loose in the current __init__. 
        # Ideally, DecisionStructure should know which set index corresponds to what.
        
        # For this implementation, we assume a strict ordering matching __init__:
        # 1. Binary sets (flattened into binary_dim bits)
        # 2. Permutation sets (in order)
        # 3. Continuous sets (flattened)
        
        batch_parts = []
        n_sols = len(solutions)
        
        # We need to map from "list of lists" to flat vectors.
        # Since DecisionStructure doesn't store "which index is which set type",
        # we have to rely on the caller or external knowledge.
        # BUT, the usage in GA will likely pass the whole 'population'.
        # Let's assume the user of DecisionStructure knows how to map.
        # Wait, that's dangerous.
        
        # Better: decode_raw_output assumed a structure.
        # encode_solution must inverse it.
        # But 'Solution' stores values, not one-hot.
        
        # Let's try to infer or require the separate parts. No, that makes integration hard.
        # Let's update `__init__` to store `set_config`?
        
        # Assuming provided solutions are LISTS of integer/double arrays for now?
        # No, the method signature is `encode_solution(self, solutions)`.
        # Taking `Solution` objects.
        
        # To make this robust, we should probably pass the data in parts if we can't easily map `Solution` to structure.
        # OR, we assume `solutions` is a list of objects that we extract data from strictly.
        
        # Let's assume strict structure:
        # int_values[0] -> binary bits for first binary segment?
        # This is getting complicated because `Solution` clumps all int lists together.
        
        # Simplification for integration:
        # We will parse `Solution` objects into flat lists first based on metadata passed to `_train_neural_models`,
        # then pass those flat lists here?
        # No, `encode_solution` should be the helper.
        
        # Let's just implement `encode_vectors` taking explicit parts:
        # binary_vecs (N, bin_dim), perm_indices (list of (N, k)), cont_vecs (N, cont_dim).
        pass

    def encode_from_parts(self, binary_part=None, perm_parts=None, cont_part=None, device='cpu'):
        """
        Encodes explicit data parts into the flat vector v.
        
        Args:
            binary_part (Tensor, optional): (N, binary_dim) - 0/1 values
            perm_parts (list of LongTensor, optional): List of (N, k) indices for each permutation set.
            cont_part (Tensor, optional): (N, continuous_dim) - real values
        """
        parts = []
        batch_size = 0
        if binary_part is not None:
             parts.append(binary_part.float())
             batch_size = binary_part.shape[0]
             
        if perm_parts:
            if batch_size == 0: batch_size = perm_parts[0].shape[0]
            for i, (n, k) in enumerate(self.permutation_dims):
                indices = perm_parts[i] # (N, k)
                # Convert to one-hot (N, k, n)
                # indices values must be in [0, n-1]
                one_hot = F.one_hot(indices.long(), num_classes=n).float()
                # Flatten -> (N, k*n)
                parts.append(one_hot.reshape(batch_size, -1))
        
        if cont_part is not None:
            if batch_size == 0: batch_size = cont_part.shape[0]
            parts.append(cont_part.float())
            
        return torch.cat(parts, dim=1).to(device)

class VAE(nn.Module):
    def __init__(self, structure, latent_dim=32, beta=1.0):
        super(VAE, self).__init__()
        self.structure = structure
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_dim = structure.total_dim
        
        # Encoder
        self.enc_fc1 = nn.Linear(self.input_dim, 512)
        self.enc_bn1 = nn.BatchNorm1d(512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_bn2 = nn.BatchNorm1d(256)
        self.enc_fc3 = nn.Linear(256, 128)
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_fc3 = nn.Linear(256, 512)
        self.dec_output = nn.Linear(512, self.input_dim)
        
    def encode(self, x):
        h = F.gelu(self.enc_bn1(self.enc_fc1(x)))
        h = F.gelu(self.enc_bn2(self.enc_fc2(h)))
        h = F.gelu(self.enc_fc3(h))
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.gelu(self.dec_fc1(z))
        h = F.gelu(self.dec_fc2(h))
        h = F.gelu(self.dec_fc3(h))
        raw_out = self.dec_output(h)
        return self.structure.decode_raw_output(raw_out)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss typed per segment
        recon_loss = 0
        batch_size = x.shape[0]
        
        # Binary - BCE
        if self.structure.binary_dim > 0:
            target_bin = x[:, :self.structure.binary_dim]
            recon_bin = recon_x[:, :self.structure.binary_dim]
            # recon_bin is already sigmoid output. Use BCELoss.
            recon_loss += F.binary_cross_entropy(recon_bin, target_bin, reduction='sum')
            
        # Permutation - Categorical CE
        # x is one-hot
        if self.structure.permutation_dims:
            for i, (n, k) in enumerate(self.structure.permutation_dims):
                start = self.structure.perm_starts[i]
                end = self.structure.perm_ends[i]
                
                target_perm = x[:, start:end]
                recon_perm = recon_x[:, start:end]
                
                # Reshape to (batch*k, n) for CrossEntropy
                # target is one-hot, convert to class indices
                target_reshaped = target_perm.reshape(batch_size, k, n) 
                recon_reshaped = recon_perm.reshape(batch_size, k, n)
                
                # We need input as logits for CrossEntropy usually, but here recon is softmax probabilities?
                # Actually, decode_raw_output returns Softmax. 
                # For numerical stability it's better to return logits and use CrossEntropyLossWithLogits equivalent.
                # But to stick to the exact architecture "Typed decoding heads", I returned probs.
                # Let's just use NLLLoss on log(probs).
                
                recon_log_probs = torch.log(recon_perm.reshape(batch_size * k, n) + 1e-10)
                # target indices
                target_indices = torch.argmax(target_reshaped, dim=2).reshape(-1)
                
                recon_loss += F.nll_loss(recon_log_probs, target_indices, reduction='sum')

        # Continuous - MSE
        if self.structure.continuous_dim > 0:
            start = self.structure.binary_dim + self.structure.total_perm_dim
            target_cont = x[:, start:]
            recon_cont = recon_x[:, start:]
            recon_loss += F.mse_loss(recon_cont, target_cont, reduction='sum')

        # KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kld, recon_loss, kld

class Generator(nn.Module):
    def __init__(self, structure, cond_dim, latent_dim=64):
        super(Generator, self).__init__()
        self.structure = structure
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        self.input_dim = latent_dim + cond_dim
        self.output_dim = structure.total_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, self.output_dim)

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = F.leaky_relu(self.fc3(h), 0.2)
        raw_out = self.fc_out(h)
        return self.structure.decode_raw_output(raw_out)

class Discriminator(nn.Module):
    def __init__(self, structure, cond_dim):
        super(Discriminator, self).__init__()
        self.input_dim = structure.total_dim + cond_dim
        
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, decision_vector, labels):
        x = torch.cat([decision_vector, labels], dim=1)
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = F.leaky_relu(self.fc3(h), 0.2)
        return self.fc_out(h)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels, device='cpu'):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    alpha = alpha.expand(real_samples.size()) # Broadcast to shape of samples
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates, labels)
    
    fake = torch.autograd.Variable(torch.ones(d_interpolates.shape), requires_grad=False).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
