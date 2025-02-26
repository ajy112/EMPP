import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_max
from torch_scatter.composite import scatter_softmax

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from IPython import embed

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from .gaussian_rbf import GaussianRadialBasisLayer

from .tools import RadialBasis, ToS2Grid_block
# for bessel radial basis
# from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis


_RESCALE = True
_USE_BIAS = True

# QM9
# _MAX_ATOM_TYPE = 5
# MD17
_MAX_ATOM_TYPE = 64
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666
  

def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))
    

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)
            

def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out
        

class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization, norm_layer=norm_layer)
        self.gate = gate
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out
    

class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp    


class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='graph', 
        internal_weights=False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)
        
        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()
        
        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i,  1)))
        irreps_max = o3.Irreps(irreps_max)
        
        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls
    
    
    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l                
        
    
    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1
    
    
    def forward(self, feature_1, feature_2):
        
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output
    
    
    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
            self.irreps_1, self.irreps_2)

        
@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)
        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        if self.nonlinear_message:          
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
                    

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output
    

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -3.0,
        stop: float = 3.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:

        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''
    
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer='layer',
        basis_type='gaussian'):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        
        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr, 
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head, 
            num_heads=self.num_heads, 
            irreps_pre_attn=self.irreps_pre_attn, 
            rescale_degree=self.rescale_degree, 
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop, 
            proj_drop=proj_drop)
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        
        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        #self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input, 
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input, #self.concat_norm_output.irreps_out, 
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output, 
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr, 
                self.irreps_node_output, 
                bias=True, rescale=_RESCALE)
        
        # add target_embed
        # self.num_gaussians = 64
        # self.e_embed = nn.Sequential(GaussianSmearing(num_gaussians=self.num_gaussians), 
        #     LinearRS(o3.Irreps('{}x0e'.format(self.num_gaussians)), 
        #     self.irreps_node_input)
        #     )

        # node-wise embedding
        # self.num_gaussians = 64
        self.num_gaussians = 64
        # self.e_embed = nn.Sequential(GaussianSmearing(num_gaussians=self.num_gaussians), 
        #     LinearRS(o3.Irreps('{}x0e'.format(self.num_gaussians)),
        #     o3.Irreps('32x0e'))
        #     )
        # self.e_embed = LinearRS(o3.Irreps('32x0e'))
        self.f_embed = LinearRS(o3.Irreps('1x0e+1x1e+1x2e+1x3e'), self.irreps_node_input)
        self.atom_embed = NodeEmbeddingNetwork(o3.Irreps('32x0e'),  _MAX_ATOM_TYPE)
        self.act_embed = torch.nn.SiLU()
        self.fc_atom_embed = LinearRS(o3.Irreps('32x0e'), self.irreps_node_input)
        
        # node-wise embedding
        # self.f_embed = TargetEdgeEmbedding(self.irreps_node_input, self.irreps_edge_attr, fc_neurons, basis_type)
        # self.m_embed = TargetEdgeEmbedding(self.irreps_node_input, self.irreps_edge_attr, fc_neurons, basis_type)
        
            
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, target, **kwargs):

        node_output = node_input

        if target is not None:
            e, f_mask, mask_atom = target
            f_mask = o3.spherical_harmonics([0,1,2,3], f_mask, normalize=False)
            f_embed = self.f_embed(f_mask)
            # e_embed = self.e_embed(e)
            mask_atom, _, _ = self.atom_embed(mask_atom)
            mask_atom_f_mask = mask_atom# + e_embed
            e_mask_embed = self.act_embed(mask_atom_f_mask)
            e_mask_embed = self.fc_atom_embed(e_mask_embed)
            e_mask_embed = e_mask_embed + f_embed
            e_mask_embed = torch.index_select(e_mask_embed, 0, batch)
            node_input = node_input + e_mask_embed

            # node_input = node_input + target_embed[batch]

        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        #norm_1_output = node_features
        node_features = self.ga(node_input=node_features, 
            node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, 
            edge_attr=edge_attr, edge_scalars=edge_scalars,
            batch=batch)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        #node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features
        
        return node_output

class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )

class TargetEdgeEmbedding(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, basis_type):
        super().__init__()
        self.basis_type = basis_type
        self.irreps_node_embedding = irreps_node_embedding
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)

        self.rbf = GaussianRadialBasisLayer(32, cutoff=5.0)
        # if self.basis_type == 'gaussian':
        #     self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        # elif self.basis_type == 'bessel':
        #     self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
        #         rbf={'name': 'spherical_bessel'})
        # elif self.basis_type == 'exp':
        #     self.rbf = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=5.0, 
        #         num_rbf=32, trainable=False)
        # else:
        #     raise ValueError
        
    
    def forward(self, x):
        weight = self.rbf(x.norm(dim=1))
        weight = self.rad(weight)
        features = o3.spherical_harmonics(l=torch.arange(self.irreps_node_embedding.lmax + 1).tolist(), x=x, normalize=True, normalization='component')
        # features = self.proj(features)
        one_features = torch.ones_like(x.narrow(1, 0, 1))
        one_features = self.exp(one_features)
        features = self.dw(one_features, features, weight)
        features = self.proj(features)

        return features
        
class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    

class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features
    

class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
        irreps_in='5x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64], 
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=False,
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        norm_layer='layer',
        alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
        drop_path_rate=0.0,
        mean=None, std=None, scale=None, atomref=None):
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, 
                rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)
        
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()
        
        self.fc_inv = LinearRS(self.irreps_node_embedding, self.irreps_feature, rescale=_RESCALE)
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE), 
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE)) 
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)
        
        self.norm_position = get_norm_layer(self.norm_layer)(self.irreps_node_embedding)
        self.postion_prediction = pos_prediction(self.irreps_feature, self.norm_position, self.irreps_node_embedding, self.atom_embed)

        self.apply(self._init_weights)
        
    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                # irreps_block_output = self.irreps_feature
                irreps_block_output = self.irreps_node_embedding
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, 
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons, 
                irreps_head=self.irreps_head, 
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn, 
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop, 
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer)
            self.blocks.append(blk)
            
            
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
                          
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
        

    def forward(self, f_in, pos, batch, node_atom, mask_node_features=None, target=None, mask_atom=None, mask_position=None, node_head=False, **kwargs) -> torch.Tensor:
        
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
            max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        
        node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)

        edge_length = edge_vec.norm(dim=1)
        #edge_length_embedding = sin_pos_embedding(x=edge_length, 
        #    start=0.0, end=self.max_radius, number=self.number_of_basis, 
        #    cutoff=False)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        # node_orifeature = node_features
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
        
        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, 
                batch=batch, target=target)

        # if 
        if not node_head:
            node_inv_features = self.fc_inv(node_features)
            node_inv_features = self.norm(node_inv_features, batch=batch)
            if self.out_dropout is not None:
                node_inv_features = self.out_dropout(node_inv_features)
            outputs = self.head(node_inv_features)
            outputs = self.scale_scatter(outputs, batch, dim=0)
            
            if self.scale is not None:
                outputs = self.scale * outputs
            
            return outputs, node_features
        else:
            position_logit, center, length_features, label_pos_length = self.postion_prediction(pos, node_features, mask_node_features, mask_atom, mask_position, batch, target)
            return position_logit, center, length_features, label_pos_length

class pos_prediction(torch.nn.Module):
    def __init__(self, irreps_feature, norm, irreps_node_embedding, atom_embed=None, max_atom_type=_MAX_ATOM_TYPE, res=100):
        super().__init__()
        self.temperature = 0.01
        self.act = nn.SiLU()
        self.norm_position = norm

        self.irreps_node_embedding = irreps_node_embedding
        self.irreps_feature = irreps_feature

        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_embedding)
        self.mid_irreps = irreps_scalars + irreps_gates + irreps_gated
        self.mid_irreps = self.mid_irreps.simplify()

        self.fc_focus1 = LinearRS(self.irreps_node_embedding, self.mid_irreps, rescale=_RESCALE)
        self.gate1 = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )

        self.fc_focus2 = LinearRS(self.irreps_node_embedding, o3.Irreps('1x0e'), rescale=_RESCALE)
        self.atom_embed = atom_embed

        self.fc_position = nn.Sequential(LinearRS(self.irreps_node_embedding, self.mid_irreps, rescale=_RESCALE),
            # Activation(o3.Irreps('64x0e+64x1e+64x2e'), acts=[torch.nn.SiLU()]),
            Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            ),
            LinearRS(self.irreps_node_embedding, o3.Irreps('16x0e+16x1e+16x2e'), rescale=_RESCALE)
        )

        # self.fc_length = LinearRS(self.irreps_node_embedding, o3.Irreps('1x0e'), rescale=_RESCALE)

        self.average_block = average(self.irreps_node_embedding)
        self.reshape_block = reshape(o3.Irreps('16x0e+16x1e+16x2e'))
        # self.mul = e3nn.o3.ElementwiseTensorProduct(o3.Irreps('1x0e+1x1e+1x2e'), o3.Irreps('3x0e'))
        self.s2 = ToS2Grid_block(self.irreps_node_embedding.lmax, res)

        # energy embedding
        self.num_gaussians = 64
        self.t_embed = GaussianSmearing(num_gaussians=self.num_gaussians)
        self.t_fc = LinearRS(o3.Irreps('{}x0e'.format(self.num_gaussians)), 
            self.irreps_node_embedding)

    def forward(self, pos, node_features, mask_node_features, mask_atom, mask_position, batch, target):
        t_embed = self.act(self.t_fc(self.t_embed(target)))
        node_embed = mask_node_features + t_embed
        node_embed = node_embed[batch]

        node_embed = self.norm_position(node_features + node_embed)
        focus_features = self.fc_focus1(node_embed)
        focus_features = self.gate1(focus_features)
        focus_logit = self.fc_focus2(focus_features)
        focus_logit = scatter_softmax(focus_logit, batch, dim=0)
        # focus = scatter_max(focus_logit_max, batch)[1]
        
        focus_features = focus_logit * focus_features

        focus_features = scatter(focus_features, batch, dim=0)
        
        position_features = self.fc_position(focus_features)
        # focus_out = self.average_block(focus_features)

        position_out = self.reshape_block(position_features)
        position_logit = self.s2.ToGrid(position_out)
        position_logit = position_logit.reshape(position_logit.shape[0], position_logit.shape[1], -1)
        position_logit = position_logit - position_logit.max(dim=1, keepdim=True)[0]
        position_logit = torch.log(torch.exp(position_logit).sum(dim=1))
        # position_logit = position_logit
        res = torch.softmax(position_logit / self.temperature, -1)

        center = scatter(pos * focus_logit, batch, dim=0)
        label_pos = mask_position - center
        label_pos_length = label_pos.norm(dim=1, keepdim=True)
        label_pos = o3.spherical_harmonics([0, 1, 2], label_pos, False)
        label_logit = self.s2.ToGrid(label_pos.unsqueeze(1))
        label_logit = label_logit.reshape(label_logit.shape[0], -1)
        label_logit = torch.softmax(label_logit / self.temperature, -1)
        # center = pos[focus]

        length_features = None
        # length_features = self.fc_length(focus_features)


        return res, label_logit.detach(), length_features, label_pos_length.detach()
        # focus

class reshape(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps
    
    def forward(self, x):

        ix = 0
        out = torch.tensor([], dtype=x.dtype, device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, ix: ix + mul * d]
            ix = ix + mul * d

            field = field.reshape(-1, mul, d)
            out = torch.cat([out, field], dim=-1)
        return out

class average(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps
    
    def forward(self, x):

        ix = 0
        out = torch.tensor([], device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, ix: ix + mul * d]
            ix = ix + mul * d

            field = field.reshape(-1, mul, d)
            out = torch.cat([out, field.mean(1)], dim=-1)
        return out

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

# @register_model
# def graph_attention_transformer_l2(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], 
#         irreps_feature='512x0e',
#         irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=False,
#         irreps_mlp_mid='384x0e+192x1e+96x2e',
#         norm_layer='layer',
#         alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model


# @register_model
# def graph_attention_transformer_nonlinear_l2(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], 
#         irreps_feature='512x0e',
#         irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=True,
#         irreps_mlp_mid='384x0e+192x1e+96x2e',
#         norm_layer='layer',
#         alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model


# @register_model
# def graph_attention_transformer_nonlinear_l2_e3(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+32x0o+32x1e+32x1o+16x2e+16x2o', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1o+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], 
#         irreps_feature='512x0e',
#         irreps_head='32x0e+8x0o+8x1e+8x1o+4x2e+4x2o', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=True,
#         irreps_mlp_mid='384x0e+96x0o+96x1e+96x1o+48x2e+48x2o',
#         norm_layer='layer',
#         alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model


# # Equiformer, L_max = 2, Bessel radial basis, dropout = 0.2
# @register_model
# def graph_attention_transformer_nonlinear_bessel_l2(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
#         irreps_feature='512x0e',
#         irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=True,
#         irreps_mlp_mid='384x0e+192x1e+96x2e',
#         norm_layer='layer',
#         alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model


# # Equiformer, L_max = 2, Bessel radial basis, dropout = 0.1
# @register_model
# def graph_attention_transformer_nonlinear_bessel_l2_drop01(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
#         irreps_feature='512x0e',
#         irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=True,
#         irreps_mlp_mid='384x0e+192x1e+96x2e',
#         norm_layer='layer',
#         alpha_drop=0.1, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model


# # Equiformer, L_max = 2, Bessel radial basis, dropout = 0.0
# @register_model
# def graph_attention_transformer_nonlinear_bessel_l2_drop00(irreps_in, radius, num_basis=128, 
#     atomref=None, task_mean=None, task_std=None, **kwargs):
#     model = GraphAttentionTransformer(
#         irreps_in=irreps_in,
#         irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
#         irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
#         max_radius=radius,
#         number_of_basis=num_basis, fc_neurons=[64, 64], basis_type='bessel',
#         irreps_feature='512x0e',
#         irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
#         rescale_degree=False, nonlinear_message=True,
#         irreps_mlp_mid='384x0e+192x1e+96x2e',
#         norm_layer='layer',
#         alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
#         mean=task_mean, std=task_std, scale=None, atomref=atomref)
#     return model