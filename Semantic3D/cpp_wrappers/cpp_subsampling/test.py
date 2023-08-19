from models.blocks import *
import numpy as np


class KPCNN(nn.Module):
    def __init__(self, config):
        super(KPCNN, self).__init__()
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius  # 卷积半径随着网格大小变化
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()
        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')
            # Detect upsampling block to stop
            if 'upsample' in block:
                break
            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config))
            # Index of block in this layer
            block_in_layer += 1
            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim
            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0
        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)
        ################
        # Network Losses
        ################
        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):
        # Save all block operations in a list of modules
        x = batch.features.clone().detach()
        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)
        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)
        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total
