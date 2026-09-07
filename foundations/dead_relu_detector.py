import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []

        def hook_fn(module, input, output):
            # output shape: [batch_size, num_neurons] (or [batch_size, channels, ...])
            # A neuron is dead if it outputs 0 across ALL samples in the batch.
            with torch.no_grad():
                # Flatten all dimensions except the neuron/feature dimension (dim 1)
                if output.ndim > 2:
                    # e.g., Conv2D output: [B, C, H, W] -> [B, C, H*W]
                    output_flat = output.flatten(2)
                    # Max per sample and spatial location to check if feature map ever activated
                    max_per_neuron = output_flat.abs().sum(dim=(0, 2))
                else:
                    # Linear layer output: [B, N]
                    # Sum across batch dimension (dim 0)
                    max_per_neuron = output.abs().sum(dim=0)
                
                dead_neurons = (max_per_neuron == 0).sum().item()
                total_neurons = max_per_neuron.numel()
                
                fraction = round(float(dead_neurons / total_neurons), 4)
                dead_fractions.append(fraction)

        hooks = []
        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        for layer in dead_fractions:
            if layer > 0.5:
                return "use_leaky_relu"
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        
        for i in range(1,len(dead_fractions)):
            curr = dead_fractions[i]
            prev = dead_fractions[i-1]
            if curr < prev:
                return "healthy"
        if dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"
        else:
            return "healthy"

