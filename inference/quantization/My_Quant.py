import torch
from torch import nn

def mantissa_clip(x, mbit):
    mantissa, exponent = torch.frexp(x)
    mantissa = torch.round(mantissa * (2<<mbit)) / (2<<mbit) 
    y = torch.ldexp(mantissa, exponent).to(x.dtype)
    return y

def abs_clip(y, clip_0, clip_1):
    z = torch.zeros_like(y)
    y = torch.where(torch.abs(y) < clip_0, z, y) 
    y = torch.max(y, torch.tensor(-clip_1, device=y.device)) 
    y = torch.min(y, torch.tensor(clip_1, device=y.device)) 
    return y

def A_quant(x, n_bits=8, range_max=None, m = 0):
    device, dtype = x.device, x.dtype
    if isinstance(n_bits, str) and n_bits == 'fp32':
        return x
    elif isinstance(n_bits, str) and n_bits == 'fp16': 
        return (x.half()).to(dtype=dtype)
    elif isinstance(n_bits, str) and n_bits == 'fp8':
        mbit = 4
        bias = 5
        ebit = 7 - mbit
        clip_0 = pow(2, -1*pow(2, ebit-1)-bias+1)
        clip_1 = (pow(2, mbit+1)-1)/pow(2, mbit)*pow(2, pow(2, ebit-1)-bias)
        if range_max is None:
            clip_max = x.abs().max()
        else:
            clip_max = range_max
        delta = torch.tensor(clip_1, device=device, dtype=dtype) / torch.max(clip_max, torch.tensor(clip_0/4, device=device, dtype=dtype))
        y = x * delta
        y = mantissa_clip(y, mbit)
        y = abs_clip(y, clip_0, clip_1)
        y = y / delta
        return y
    elif m > 0:
        qmax, qmin = 2**(n_bits-2)-1, -(2**(n_bits-2))+1
        x_min, x_max = -range_max, range_max
        delta = torch.max((x_max - x_min) / (qmax - qmin), torch.tensor(1e-5, device=device, dtype=dtype))
        delta_small = delta / 2.0**m
        x_tmp = (x / delta_small).round()
        x1 = x.clone()
        x1 = x1.masked_fill_((x_tmp > qmax) | (x_tmp < qmin), 0)
        x1_quant = torch.clamp((x1 / delta_small).round(), qmin, qmax)
        x1_dequant = x1_quant * delta_small
        x2 = x.clone()
        x2 = x2.masked_fill_((x_tmp <= qmax) & (x_tmp >= qmin), 0)
        x2_quant = torch.clamp((x2 / delta).round(), qmin, qmax)
        x2_dequant = x2_quant * delta
        return x1_dequant + x2_dequant
    else: #目前仅支持对称量化
        qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
        if range_max is None:
            x_absmax = x.abs().max()
            x_min, x_max = -x_absmax, x_absmax
        else:
            x_min, x_max = -range_max, range_max
        delta = torch.max((x_max - x_min) / (qmax - qmin), torch.tensor(1e-5, device=device, dtype=dtype))
        x_quant = torch.clamp((x / delta).round(), qmin, qmax)
        x_dequant = x_quant * delta
        return x_dequant

def A_VSQuant(x, n_bits, V, trans):
    import math
    device, dtype = x.device, x.dtype
    qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
    if trans is False:
        K = x.shape[-1]
        V_num = math.ceil(float(K) / float(V))
        for i in range(V_num):
            if i < (V_num - 1):
                x_tiled = x[:,i*V:(i+1)*V]
            else:
                x_tiled = x[:,i*V:]
            x_absmax, _ = x_tiled.abs().max(dim=-1)
            x_min, x_max = -x_absmax, x_absmax
            delta = torch.max((x_max - x_min) / (qmax - qmin), torch.tensor(1e-5, device=device, dtype=dtype))
            delta = delta.unsqueeze(-1)
            x_quant = torch.clamp((x_tiled / delta).round(), qmin, qmax)
            x_dequant = x_quant * delta
            if i < (V_num - 1):
                x[:,i*V:(i+1)*V] = x_dequant
            else:
                x[:,i*V:] = x_dequant
    else:
        K = x.shape[-2]
        V_num = math.ceil(float(K) / float(V))
        for i in range(V_num):
            if i < (V_num - 1):
                x_tiled = x[:,i*V:(i+1)*V,:]
            else:
                x_tiled = x[:,i*V:,:]
            x_absmax, _ = x_tiled.abs().max(dim=-2)
            x_min, x_max = -x_absmax, x_absmax
            delta = torch.max((x_max - x_min) / (qmax - qmin), torch.tensor(1e-5, device=device, dtype=dtype))
            delta = delta.unsqueeze(-2)
            x_quant = torch.clamp((x_tiled / delta).round(), qmin, qmax)
            x_dequant = x_quant * delta
            if i < (V_num - 1):
                x[:,i*V:(i+1)*V,:] = x_dequant
            else:
                x[:,i*V:,:] = x_dequant
    return x

class STE(nn.Module):  # 用于自定义需要插入数据量化的节点
    def __init__(self, trans=False):
        super().__init__()
        self.n_bits = None
        self.k = None #用于smooth quant
        self.V = None #用于VSQuant
        self.trans = trans
        self.m = 0 

    def forward(self, x):
        if self.k is None:
            if self.V is not None and self.n_bits is not None:
                return A_VSQuant(x, self.n_bits, self.V, self.trans)
            elif hasattr(self, "outputmaxminhook") and (self.n_bits is not None):
                return A_quant(x, self.n_bits, torch.max(self.outputmaxminhook.output_min.abs(), self.outputmaxminhook.output_max.abs()).to(x.device), self.m)
            elif self.n_bits is not None:
                return A_quant(x, self.n_bits)
            else:
                return x
        else:
            if self.V is not None and self.n_bits is not None:
                return A_VSQuant(torch.mul(x, self.k.to(x.device)), self.n_bits, self.V, self.trans)
            elif hasattr(self, "outputmaxminhook") and (self.n_bits is not None):
                return A_quant(torch.mul(x, self.k.to(x.device)), self.n_bits, torch.max(self.outputmaxminhook.output_min.abs(), self.outputmaxminhook.output_max.abs()).to(x.device), self.m)
            elif self.n_bits is not None:
                return A_quant(torch.mul(x, self.k.to(x.device)), self.n_bits)
            else:
                return torch.mul(x, self.k.to(x.device))

class My_MatMul(torch.nn.Module): #用于替换torch.matmul
    def __init__(self):
        super().__init__()
        self.ste0 = STE()
        self.ste1 = STE(trans=True)

    def forward(self, x, y):
        return torch.matmul(self.ste0(x), self.ste1(y))
    
class My_QuantModule(torch.nn.Module):  #用于替换torch.linear、conv等
    def __init__(self, org_module):
        super().__init__()
        self.org_module = org_module
        self.ste0 = STE()
        self.n_bits = None
        self.squant_k = True
        self.squant_c = True
        self.percdamp = 0.01
        self.blocksize = 128
        self.W_sys = True
        self.actorder = True
        self.groupsize = 64 #64 #-1

    def forward(self, x):
        return self.org_module(self.ste0(x))

    def W_UniformQuant(self):
        device = self.org_module.weight.device
        n_bits = self.n_bits
        if isinstance(n_bits, str) and n_bits == 'fp32':
            pass
        elif isinstance(n_bits, str) and n_bits == 'fp16': 
            dtype = self.org_module.weight.data.dtype
            self.org_module.weight.data = (self.org_module.weight.data.half()).to(dtype)
        elif isinstance(n_bits, str) and n_bits == 'fp8':
            mbit = 4
            bias = 5
            ebit = 7 - mbit
            clip_0 = pow(2, -1*pow(2, ebit-1)-bias+1)
            clip_1 = (pow(2, mbit+1)-1)/pow(2, mbit)*pow(2, pow(2, ebit-1)-bias)
            clip_max = self.org_module.weight.data.abs().max()
            delta = torch.tensor(clip_1, device=device) / torch.max(clip_max, torch.tensor(clip_0/4, device=device))
            y = self.org_module.weight.data * delta
            y = mantissa_clip(y, mbit)
            y = abs_clip(y, clip_0, clip_1)
            y = y / delta
            self.org_module.weight.data = y
        else:
            qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
            W_absmax, _ = self.org_module.weight.data.abs().max(dim=-1)
            W_min, W_max = -W_absmax, W_absmax
            delta = torch.max((W_max - W_min) / (qmax - qmin), torch.tensor(1e-5, device=device))
            delta = delta.unsqueeze(-1)
            W_quant = torch.clamp((self.org_module.weight.data / delta).round(), qmin, qmax)
            W_dequant = W_quant * delta
            self.org_module.weight.data = W_dequant

    def rounding_forward(self,
                         rounding_error_sum, rounding_number_, rounding_error_,
                         number_, error_, priority_, order_,
                         priority_1):
        topk = rounding_error_sum.abs().round()
        over_squant = topk >= rounding_error_sum.abs()
        topk = int(topk)
        if topk > 0:
            rounding_error_[order_[0:topk]] = error_[order_[0:topk]]
            rounding_number_[order_[0:topk]] = number_[order_[0:topk]]
            if over_squant:
                idx_c = order_[topk - 1]
                priority_1[idx_c] = rounding_error_[idx_c].abs()
            else:
                idx_c = order_[topk]
                priority_[idx_c] = rounding_error_[idx_c].abs()
        return rounding_number_, rounding_error_, priority_, priority_1

    def SQuant_func(self,
                    rounding_error_sum, rounding_number, rounding_error,
                    up_number, up_error, up_priority, up_order,
                    down_number, down_error, down_priority, down_order):
        rounding_number_shape = rounding_number.shape
        batch_size = rounding_number_shape[0]
        input_channel = rounding_number_shape[1]
        for n in range(batch_size):
            for c in range(input_channel):
                if rounding_error_sum[n, c] < 0:
                    rounding_number[n, c], rounding_error[n, c], up_priority[n, c], down_priority[n, c] \
                        = self.rounding_forward(rounding_error_sum[n, c],
                                                rounding_number[n, c],
                                                rounding_error[n, c],
                                                up_number[n, c],
                                                up_error[n, c],
                                                up_priority[n, c],
                                                up_order[n, c],
                                                down_priority[n, c])
                else:
                    rounding_number[n, c], rounding_error[n, c], down_priority[n, c], up_priority[n, c] \
                        = self.rounding_forward(rounding_error_sum[n, c],
                                                rounding_number[n, c],
                                                rounding_error[n, c],
                                                down_number[n, c],
                                                down_error[n, c],
                                                down_priority[n, c],
                                                down_order[n, c],
                                                up_priority[n, c])
        return rounding_number, rounding_error, up_priority, down_priority

    def adaptive_round(self, x, t_min=None, t_max=None):
        rounding_number = x.round()  # round取整值
        rounding_error = rounding_number - x  # 误差

        up_number = rounding_number.clone()
        up_error = rounding_error.clone()
        up_error[x >= t_max] = 0.0  # 边界上的值不能再调整，所以去除
        up_error[up_error > 0] = 0.0  # 误差为正的都设为0，即up对应“原值>量化值”的集合
        up_priority = up_error.clone().abs()

        up_error[up_error != 0] += 1  # up集合中，Flip翻转后对应的误差
        up_number[up_error != 0] += 1  # up集合中，Flip翻转后对应的取整值

        down_number = rounding_number.clone()
        down_error = rounding_error.clone()
        down_error[x <= t_min] = 0.0  # 边界上的值不能再调整，所以去除
        down_error[down_error < 0] = 0.0  # 误差为负的都设为0，即down对应“原值<量化值”的集合
        down_priority = down_error.clone().abs()

        down_error[down_error != 0] -= 1  # down集合中，Flip翻转后对应的误差
        down_number[down_error != 0] -= 1  # down集合中，Flip翻转后对应的取整值

        conver_shape = x.view(x.shape[0], x.shape[1], -1).shape  # HW维度合并
        if conver_shape[2] == 1:
            self.squant_k = False  # 只有一个元素时， 不做K的逼近
        if self.squant_k:
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
            _, up_order = torch.sort(up_priority.view(
                conver_shape), descending=True)  # up集合误差从大到小排序
            _, down_order = torch.sort(down_priority.view(
                conver_shape), descending=True)  # down集合误差从大到小排序
            up_priority *= 0.0
            down_priority *= 0.0

            rounding_number, rounding_error, up_priority, down_priority = self.SQuant_func(
                rounding_error_sum,
                rounding_number.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order,

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order,
            )
            rounding_number = rounding_number.view(x.shape)
            rounding_error = rounding_error.view(x.shape)
            up_priority = up_priority.view(x.shape)
            down_priority = down_priority.view(x.shape)

        if self.squant_c:
            conver_shape = (1, x.shape[0], -1)
            rounding_error_sum = rounding_error.view(conver_shape).sum(-1)
            _, up_order = torch.sort(
                up_priority.view(conver_shape), descending=True)
            _, down_order = torch.sort(
                down_priority.view(conver_shape), descending=True)

            rounding_number, rounding_error, up_priority, down_priority = self.SQuant_func(
                rounding_error_sum,
                rounding_number.view(conver_shape),
                rounding_error.view(conver_shape),

                up_number.view(conver_shape),
                up_error.view(conver_shape),
                up_priority.view(conver_shape),
                up_order,

                down_number.view(conver_shape),
                down_error.view(conver_shape),
                down_priority.view(conver_shape),
                down_order
            )
            rounding_number = rounding_number.view(x.shape)
            rounding_error = rounding_error.view(x.shape)
            up_priority = up_priority.view(x.shape)
            down_priority = down_priority.view(x.shape)

        return rounding_number

    def W_SQuant(self):
        device = self.org_module.weight.device
        n_bits = self.n_bits
        if isinstance(n_bits, str) and n_bits == 'fp32':
            pass
        elif isinstance(n_bits, str) and n_bits == 'fp16': 
            dtype = self.org_module.weight.data.dtype
            self.org_module.weight.data = (self.org_module.weight.data.half()).to(dtype)
        elif isinstance(n_bits, str) and n_bits == 'fp8':
            mbit = 4
            bias = 5
            ebit = 7 - mbit
            clip_0 = pow(2, -1*pow(2, ebit-1)-bias+1)
            clip_1 = (pow(2, mbit+1)-1)/pow(2, mbit)*pow(2, pow(2, ebit-1)-bias)
            clip_max = self.org_module.weight.data.abs().max()
            delta = torch.tensor(clip_1, device=device) / torch.max(clip_max, torch.tensor(clip_0/4, device=device))
            y = self.org_module.weight.data * delta
            y = mantissa_clip(y, mbit)
            y = abs_clip(y, clip_0, clip_1)
            y = y / delta
            self.org_module.weight.data = y
        else:
            qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
            W_absmax, _ = self.org_module.weight.data.abs().max(dim=-1)
            W_min, W_max = -W_absmax, W_absmax
            delta = torch.max((W_max - W_min) / (qmax - qmin), torch.tensor(1e-5, device=device))
            delta = delta.unsqueeze(-1)
            W_quant = self.org_module.weight.data / delta
            W_int = self.adaptive_round(W_quant, qmin, qmax)
            W_quant = torch.clamp(W_int, qmin, qmax)
            W_dequant = W_quant * delta
            self.org_module.weight.data = W_dequant

    def W_VSQuant(self):
        import math
        device = self.org_module.weight.device
        n_bits = self.n_bits
        if isinstance(n_bits, str) and n_bits == 'fp32':
            pass
        elif isinstance(n_bits, str) and n_bits == 'fp16': 
            dtype = self.org_module.weight.data.dtype
            self.org_module.weight.data = (self.org_module.weight.data.half()).to(dtype)
        elif isinstance(n_bits, str) and n_bits == 'fp8':
            mbit = 4
            bias = 5
            ebit = 7 - mbit
            clip_0 = pow(2, -1*pow(2, ebit-1)-bias+1)
            clip_1 = (pow(2, mbit+1)-1)/pow(2, mbit)*pow(2, pow(2, ebit-1)-bias)
            clip_max = self.org_module.weight.data.abs().max()
            delta = torch.tensor(clip_1, device=device) / torch.max(clip_max, torch.tensor(clip_0/4, device=device))
            y = self.org_module.weight.data * delta
            y = mantissa_clip(y, mbit)
            y = abs_clip(y, clip_0, clip_1)
            y = y / delta
            self.org_module.weight.data = y
        else:
            qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
            V = self.groupsize
            if isinstance(self.org_module, nn.Linear):
                weight = self.org_module.weight.data
            else:
                weight = self.org_module.weight.data.reshape(self.org_module.weight.data.shape[0], -1)
            K = weight.shape[-1]
            V_num = math.ceil(float(K) / float(V))
            d_max = torch.tensor(0.0, device=device)
            delta = []
            rst = torch.empty_like(weight)
            for i in range(V_num):
                if i < (V_num - 1):
                    W_tiled = weight.data[:,i*V:(i+1)*V]
                else:
                    W_tiled = weight.data[:,i*V:]
                W_absmax, _ = W_tiled.abs().max(dim=-1)
                W_min, W_max = -W_absmax, W_absmax
                delta.append(torch.max((W_max - W_min) / (qmax - qmin), torch.tensor(1e-5, device=device)))
                d_max = torch.max(d_max, delta[i].abs().max())
            d_min = -d_max
            d_n_bits = 8
            d_qmax, d_qmin = 2**(d_n_bits-1)-1, -(2**(d_n_bits-1))+1
            d_delta = torch.max((d_max - d_min) / (d_qmax - d_qmin), torch.tensor(1e-5, device=device))
            for i in range(V_num):
                if i < (V_num - 1):
                    W_tiled = weight.data[:,i*V:(i+1)*V]
                else:
                    W_tiled = weight.data[:,i*V:]
                delta_tmp = delta[i].unsqueeze(-1)
                W_quant = torch.clamp((W_tiled / delta_tmp).round(), qmin, qmax)
                delta[i] = torch.clamp((delta[i] / d_delta).round(), d_qmin, d_qmax)
                delta_tmp = delta[i].unsqueeze(-1)
                W_dequant = W_quant * (delta_tmp) * d_delta
                if i < (V_num - 1):
                    rst[:,i*V:(i+1)*V] = W_dequant
                else:
                    rst[:,i*V:] = W_dequant
            self.org_module.weight.data = rst.reshape(self.org_module.weight.data.shape)

    def RTN_quantize_scale(self, x):
        import math
        device = x.device
        n_bits = self.n_bits
        if self.W_sys:
            qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
        else:
            qmax, qmin = 2**n_bits-1, 0
        if self.groupsize != -1:
            V = self.groupsize
            K = self.org_module.weight.shape[-1]
            V_num = math.ceil(float(K) / float(V))
            d_max = torch.tensor(0.0, device=device)
            self.delta = []
            for i in range(V_num):
                if i < (V_num - 1):
                    W_tiled = x[:,i*V:(i+1)*V]
                else:
                    W_tiled = x[:,i*V:]
                W_absmax, _ = W_tiled.abs().max(dim=-1)
                W_min, W_max = -W_absmax, W_absmax
                self.delta.append(torch.max((W_max - W_min) / (qmax - qmin), torch.tensor(1e-5, device=device)))
                d_max = torch.max(d_max, self.delta[i].abs().max())
            d_min = -d_max
            d_n_bits = 8
            d_qmax, d_qmin = 2**(d_n_bits-1)-1, -(2**(d_n_bits-1))+1
            self.d_delta = torch.max((d_max - d_min) / (d_qmax - d_qmin), torch.tensor(1e-5, device=device))
        else:
            W_absmax, _ = x.abs().max(dim=-1)
            W_min, W_max = -W_absmax, W_absmax
            delta = torch.max((W_max - W_min) / (qmax - qmin), torch.tensor(1e-5, device=device))
            self.delta = delta.unsqueeze(-1)
            if self.W_sys is False:
                zero_point = torch.round((W_max * qmin - W_min * qmax) / (W_max - W_min))
                self.zero_point = zero_point.unsqueeze(-1)


    def RTN_quantize(self, W, v_idx=None):
        # normal round quantization method
        import math
        n_bits = self.n_bits
        if self.W_sys:
            qmax, qmin = 2**(n_bits-1)-1, -(2**(n_bits-1))+1
        else:
            qmax, qmin = 2**n_bits-1, 0
        if self.groupsize != -1:
            d_n_bits = 8
            d_qmax, d_qmin = 2**(d_n_bits-1)-1, -(2**(d_n_bits-1))+1
            delta_tmp = self.delta[v_idx]
            W_quant = torch.clamp((W / delta_tmp.unsqueeze(-1)).round(), qmin, qmax)
            delta_int = torch.clamp((delta_tmp / self.d_delta).round(), d_qmin, d_qmax)
            delta_tmp = delta_int * self.d_delta
            W_dequant = W_quant * delta_tmp.unsqueeze(-1)
        else:
            if self.W_sys:
                W_quant = torch.clamp((W / self.delta).round(), qmin, qmax)
                W_dequant = W_quant * self.delta
            else:
                W_quant = torch.clamp((W / self.delta).round() + self.zero_point, qmin, qmax)
                W_dequant = (W_quant - self.zero_point) * self.delta
        return W_dequant, W_quant

    def W_GPTQ(self):
        # GPTQ algorithm, details at https://arxiv.org/abs/2210.17323
        assert not isinstance(self.n_bits, str), "GPTQ not support fp8、fp16、fp32"
        device = self.org_module.weight.data.device
        if isinstance(self.org_module, nn.Linear):
            groups = 1
        else:
            groups = self.gptq_hessian_hook.groups
        rst = torch.empty_like(self.org_module.weight.data)
        for g in range(groups):
            if isinstance(self.org_module, nn.Linear):
                weight = self.org_module.weight.data
            elif isinstance(self.org_module, nn.Conv2d):
                weight = self.org_module.weight.data[g*self.gptq_hessian_hook.SizeOut_perGroup:(g+1)*self.gptq_hessian_hook.SizeOut_perGroup,:,:,:]
                weight = weight.view(weight.shape[0], -1)
            else:
                weight = self.org_module.weight.data[g*self.gptq_hessian_hook.SizeIn_perGroup:(g+1)*self.gptq_hessian_hook.SizeIn_perGroup,:,:,:].transpose(0, 1)
                weight = torch.rot90(weight, 2, [2, 3])
                weight = weight.reshape(weight.shape[0], -1)
            self.RTN_quantize_scale(weight)
            columns = weight.shape[1]
            W = weight.clone()
            W = W.float()
            if isinstance(self.org_module, nn.Linear):
                H = self.gptq_hessian_hook.H.to(device)
            else:
                H = self.gptq_hessian_hook.H[g].to(device)
            dead = torch.diag(H) == 0
            W[:, dead] = 0
            Q = torch.zeros_like(W)
            Q_int = torch.zeros_like(W)
            if not hasattr(self, "Hinv"):
                H[dead, dead] = 1
                if self.actorder:
                    perm = torch.argsort(torch.diag(H), descending=True)
                    W = W[:, perm]
                    H = H[perm][:, perm]
                    if g == 0:
                        self.perm = []
                    self.perm.append(perm.to(device='cpu'))
                damp = self.percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(columns, device=device)
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H 
                if g == 0:
                    self.Hinv = []
                self.Hinv.append(Hinv.to(device='cpu'))
            else:
                Hinv = self.Hinv[g].to(device)
                if self.actorder:
                    perm = self.perm[g].to(device)    
                    W = W[:, perm]     
            for i1 in range(0, columns, self.blocksize):
                i2 = min(i1 + self.blocksize, columns)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Q1_int = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    if self.groupsize != -1:
                        if self.actorder:
                            q, q_int = self.RTN_quantize(w.unsqueeze(1), int(perm[(i1 + i)]) // self.groupsize)
                        else:
                            q, q_int = self.RTN_quantize(w.unsqueeze(1), int(i1 + i) // self.groupsize)
                    else:
                        q, q_int = self.RTN_quantize(w.unsqueeze(1))
                    Q1[:, i] = q.flatten()
                    Q1_int[:, i] = q_int.flatten()
                    err1 = (w - q.flatten()) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
                Q[:, i1:i2] = Q1
                Q_int[:, i1:i2] = Q1_int
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            if self.actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
                Q_int = Q_int[:, invperm]
            if isinstance(self.org_module, nn.Linear):
                rst = Q.reshape(weight.shape).to(weight.dtype)
                rst_int = Q_int.reshape(weight.shape).to(weight.dtype)
                rst_int = rst_int.to(dtype=torch.int8)
                rst_int = ((rst_int[:,1::2] & 15) << 4) | (rst_int[:,0::2] & 15)
                self.org_module.weight_int = rst_int
                d_n_bits = 8
                d_qmax, d_qmin = 2**(d_n_bits-1)-1, -(2**(d_n_bits-1))+1
                delta_tmp = torch.cat(self.delta).reshape(-1,weight.shape[0]).transpose(0, 1)
                self.org_module.delta_int = torch.clamp((delta_tmp / self.d_delta).round(), d_qmin, d_qmax).to(dtype=torch.int8)
                self.org_module.s_delta = self.d_delta
            elif isinstance(self.org_module, nn.Conv2d):
                Q = Q.reshape(weight.shape).to(weight.dtype)
                shape = rst[g*self.gptq_hessian_hook.SizeOut_perGroup:(g+1)*self.gptq_hessian_hook.SizeOut_perGroup,:,:,:].shape
                rst[g*self.gptq_hessian_hook.SizeOut_perGroup:(g+1)*self.gptq_hessian_hook.SizeOut_perGroup,:,:,:] = Q.reshape(shape)
            else:
                Q = Q.reshape(weight.shape).to(weight.dtype)
                shape = rst[g*self.gptq_hessian_hook.SizeOut_perGroup:(g+1)*self.gptq_hessian_hook.SizeOut_perGroup,:,:,:].shape
                rst[g*self.gptq_hessian_hook.SizeOut_perGroup:(g+1)*self.gptq_hessian_hook.SizeOut_perGroup,:,:,:] = torch.rot90(Q.reshape(shape), 2, [2, 3])
        self.org_module.weight.data = rst

@torch.no_grad()
def get_parent_module(model, name):
    tmp_name = name.split('.')
    str_name = "model"
    for j in range(len(tmp_name)-1):
        str_name = str_name + "._modules['" + tmp_name[j] + "']"
    return eval(str_name)

@torch.no_grad()
def replace_quant_op(module, org_op_list = (), replace_op_list = ()): #搜索并替换模型中的可量化OP
    for name, child_module in module.named_children():
        # print(name, child_module)
        if isinstance(child_module, org_op_list): 
            replace_op = replace_op_list[list(org_op_list).index(child_module.__class__)]
            print(name, '(', child_module.__class__.__name__, ') is warpped by ' + replace_op.__name__ + ' Class.')
            setattr(module, name, replace_op(org_module=child_module))
        elif isinstance(child_module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)): 
            print(name, '(', child_module.__class__.__name__, ') is warpped by My_QuantModule Class.')
            setattr(module, name, My_QuantModule(org_module=child_module))
        else:
            replace_quant_op(child_module, org_op_list, replace_op_list)

class outputmaxminhook: #钩子函数，获取输出最大/最小值
    def __init__(self):
        self.output_min = None
        self.output_max = None

    @torch.no_grad()
    def __call__(self, module, input_batch, output_batch):
        if self.output_min is None:
            self.output_min = output_batch.min().to(device='cpu')
            self.output_max = output_batch.max().to(device='cpu')
        else:
            self.output_min = torch.min(
                self.output_min, output_batch.min().to(device='cpu'))
            self.output_max = torch.max(
                self.output_max, output_batch.max().to(device='cpu'))
        torch.cuda.empty_cache()

@torch.no_grad()
def add_hook_max_min_model(model: nn.Module): #对STE增加钩子函数，统计输出最大/最小值
    for name, child_module in model.named_modules():
        if isinstance(child_module, STE):
            # print(name, child_module)
            child_module.outputmaxminhook = outputmaxminhook()
            child_module.handle_max_min = child_module.register_forward_hook(
                child_module.outputmaxminhook)

@torch.no_grad()
def del_hook_max_min_model(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, STE):
            # print(name, child_module)
            child_module.handle_max_min.remove()

class smooth_hook: #钩子函数，获取smooth quant相关信息
    def __init__(self):
        self.output_smooth_min = None
        self.output_smooth_max = None

    @torch.no_grad()
    def __call__(self, module, input_batch, output_batch):
        tmp_min = output_batch.min(dim=0)[0]
        tmp_max = output_batch.max(dim=0)[0]
        if (len(output_batch.shape)-2) > 0:
            for i in range(len(output_batch.shape)-2):
                tmp_min = tmp_min.min(dim=0)[0]
                tmp_max = tmp_max.max(dim=0)[0]
        if self.output_smooth_min is None:
            self.output_smooth_min = tmp_min.to(device='cpu')
            self.output_smooth_max = tmp_max.to(device='cpu')
        else:
            self.output_smooth_min = torch.max(self.output_smooth_min, tmp_min.to(device='cpu'))
            self.output_smooth_max = torch.max(self.output_smooth_max, tmp_max.to(device='cpu'))
        torch.cuda.empty_cache()

@torch.no_grad()
def add_hook_smooth(model: nn.Module): #增加钩子函数，统计smooth quant相关信息
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule) and isinstance(child_module.org_module, nn.Linear):
            # print(name, child_module)
            child_module.ste0.smooth_hook = smooth_hook()
            child_module.ste0.handle_smooth = child_module.register_forward_hook(
                child_module.ste0.smooth_hook)

@torch.no_grad()
def del_hook_smooth(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule):
            # print(name, child_module)
            child_module.ste0.handle_smooth.remove()
            
class gptq_hessian_hook: #钩子函数，获取gptq相关信息
    def __init__(self, module):
        self.module = module.org_module
        self.nsamples = 0
        if isinstance(self.module, nn.Linear):
            columns=self.module.weight.data.shape[1]
            self.H = torch.zeros((columns, columns))
        elif isinstance(self.module, nn.Conv2d):
            columns=self.module.weight.data.shape[1]*self.module.weight.data.shape[2]*self.module.weight.data.shape[3]
            self.H = []
            for g in range(self.module.groups):
                self.H.append(torch.zeros((columns, columns)))
            self.unfold = torch.nn.Unfold(
                kernel_size=self.module.kernel_size,
                dilation=self.module.dilation,
                padding=self.module.padding,
                stride=self.module.stride
            )
            self.groups = self.module.groups
            self.SizeIn_perGroup = self.module.weight.data.shape[1]
            self.SizeOut_perGroup = self.module.weight.data.shape[0] // self.module.groups
        else:
            columns=self.module.weight.data.shape[0]//self.module.groups*self.module.weight.data.shape[2]*self.module.weight.data.shape[3]
            self.H = []
            for g in range(self.module.groups):
                self.H.append(torch.zeros((columns, columns)))
            self.unfold = torch.nn.Unfold(
                kernel_size=self.module.kernel_size,
                dilation=self.module.dilation,
                padding=(0, 0),
                stride=(1, 1)
            )
            self.groups = self.module.groups
            self.pad0 = self.module.dilation[0] * (self.module.kernel_size[0] - 1) - self.module.padding[0]
            self.pad1 = self.module.dilation[1] * (self.module.kernel_size[1] - 1) - self.module.padding[1]
            self.SizeIn_perGroup = self.module.weight.data.shape[0] // self.module.groups
            self.SizeOut_perGroup = self.module.weight.data.shape[1]
        
    @torch.no_grad()
    def __call__(self, module, inp, out):
        import math
        if isinstance(self.module, nn.Linear):
            if not hasattr(inp, 'shape'):
                inp = inp[0]
            self.H = self.H.to(device=inp.device)
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())
            self.H = self.H.to(device='cpu')
        elif isinstance(self.module, nn.Conv2d):
            if not hasattr(inp, 'shape'):
                inp = inp[0]
            tmp = inp.shape[0]
            for g in range(self.groups):
                self.H[g] = self.H[g].to(device=inp.device)
                inp_tmp = self.unfold(inp[:,g*self.SizeIn_perGroup:(g+1)*self.SizeIn_perGroup,:,:]).transpose(1, 2)
                inp_tmp = inp_tmp.reshape((-1, inp_tmp.shape[-1]))
                inp_tmp = inp_tmp.t()
                self.H[g] *= self.nsamples / (self.nsamples + tmp)
                inp_tmp = math.sqrt(2 / (self.nsamples + tmp)) * inp_tmp.float()
                self.H[g] += inp_tmp.matmul(inp_tmp.t())
                self.H[g] = self.H[g].to(device='cpu')
            self.nsamples += tmp
        else:
            if not hasattr(inp, 'shape'):
                inp = inp[0]
            tmp = inp.shape[0]
            for g in range(self.groups):
                self.H[g] = self.H[g].to(device=inp.device)
                inp_tmp = torch.zeros((inp.shape[0], self.SizeIn_perGroup,
                    self.pad0*2 + self.module.stride[0]*inp.shape[2]-(self.module.stride[0]-1),
                    self.pad1*2 + self.module.stride[1]*inp.shape[3]-(self.module.stride[1]-1)))
                inp_tmp[:,:,self.pad0:inp_tmp.shape[2]-self.pad0:self.module.stride[0],self.pad1:inp_tmp.shape[3]-self.pad1:self.module.stride[1]] = \
                    inp[:,g*self.SizeIn_perGroup:(g+1)*self.SizeIn_perGroup,:,:]
                inp_tmp = self.unfold(inp_tmp).transpose(1, 2)
                inp_tmp = inp_tmp.reshape((-1, inp_tmp.shape[-1]))
                inp_tmp = inp_tmp.t()
                self.H[g] *= self.nsamples / (self.nsamples + tmp)
                inp_tmp = math.sqrt(2 / (self.nsamples + tmp)) * inp_tmp.float()
                self.H[g] += inp_tmp.matmul(inp_tmp.t())
                self.H[g] = self.H[g].to(device='cpu')
            self.nsamples += tmp
        torch.cuda.empty_cache()
        
@torch.no_grad()
def add_hook_gptq_hessian(model: nn.Module): #对STE增加钩子函数，获取gptq相关信息
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule):
            # print(name, child_module)
            child_module.gptq_hessian_hook = gptq_hessian_hook(module=child_module)
            child_module.handle_gptq_hessian = child_module.register_forward_hook(
                child_module.gptq_hessian_hook)

@torch.no_grad()
def del_hook_gptq_hessian(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule):
            # print(name, child_module)
            child_module.handle_gptq_hessian.remove()

class histhook: #钩子函数,获取hist相关信息
    def __init__(self):
        self.hist = None

    @torch.no_grad()
    def __call__(self, module, input_batch, output_batch):
        if self.hist is None:
            self.hist = torch.histc(output_batch.to(dtype=torch.float32), bins=256, 
                                min=module.outputmaxminhook.output_min.to(dtype=torch.float32), max=module.outputmaxminhook.output_max.to(dtype=torch.float32)).to(device='cpu')
        else:
            self.hist = self.hist + torch.histc(output_batch.to(dtype=torch.float32), bins=256, 
                                min=module.outputmaxminhook.output_min.to(dtype=torch.float32), max=module.outputmaxminhook.output_max.to(dtype=torch.float32)).to(device='cpu')
        torch.cuda.empty_cache()
                    
@torch.no_grad()
def add_hook_hist(model: nn.Module): #对STE增加钩子函数，获取hist相关信息
    for name, child_module in model.named_modules():
        if isinstance(child_module, STE):
            # print(name, child_module)
            child_module.histhook = histhook()
            child_module.handle_hist = child_module.register_forward_hook(
                child_module.histhook)

@torch.no_grad()
def del_hook_hist(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, STE):
            # print(name, child_module)
            child_module.handle_hist.remove()

@torch.no_grad()
def A_get_QuantModule_num(module):
    num = 0
    QuantModule_name = []
    QuantModule_list = []
    for name, m in module.named_modules():
        if isinstance(m, (My_QuantModule, My_MatMul)):
            num = num + 1
            QuantModule_name.append(name)
            QuantModule_list.append(m)
    return num, QuantModule_name, QuantModule_list

@torch.no_grad()
def W_get_QuantModule_num(module):
    num = 0
    QuantModule_name = []
    QuantModule_list = []
    for name, m in module.named_modules():
        if isinstance(m, My_QuantModule):
            num = num + 1
            QuantModule_name.append(name)
            QuantModule_list.append(m)
    return num, QuantModule_name, QuantModule_list

@torch.no_grad()
def kl_divergence(P, Q):
    return torch.nn.functional.kl_div(P, Q, reduction='batchmean')

@torch.no_grad()
def symmetric_kl(P, Q):
    import torch.nn.functional as F
    return (kl_divergence(F.log_softmax(P, dim=-1), F.softmax(Q, dim=-1)) + kl_divergence(F.log_softmax(Q, dim=-1), F.softmax(P, dim=-1))) / 2

class A_mpqaunt_hook: #钩子函数
    def __init__(self, A_MP_bit_list, A_quant_method, A_mp_method, smooth_flag, alpha):
        super().__init__()
        self.A_MP_bit_list = A_MP_bit_list
        self.A_quant_method = A_quant_method
        self.A_mp_method = A_mp_method
        self.smooth_flag = smooth_flag
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, module, inp, out):
        import torch.nn.functional as F
        if isinstance(module, My_QuantModule):
            device, dtype = module.org_module.weight.data.device, module.org_module.weight.data.dtype
            for j, bit in enumerate(self.A_MP_bit_list):
                if ((dtype == torch.float16 or dtype == torch.half) and 'fp16' in [bit]) or 'fp32' in [bit]:
                    if not hasattr(module, 'A_Loss'):
                        module.A_Loss = []
                    if j >= len(module.A_Loss):
                        if self.A_mp_method == "MSE":
                            module.A_Loss.append(torch.tensor(1e-10, device='cpu'))
                        elif self.A_mp_method == "R2":
                            module.A_Loss.append(torch.tensor(1.0, device='cpu'))
                    else:
                        if self.A_mp_method == "MSE":
                            module.A_Loss[j] = module.A_Loss[j] + torch.tensor(1e-10, device='cpu')
                        elif self.A_mp_method == "R2":
                            module.A_Loss[j] = module.A_Loss[j] + torch.tensor(1.0, device='cpu')
                else:
                    module.ste0.n_bits = bit 
                    if self.A_quant_method == "VSQuant" and isinstance(bit, int) and isinstance(module.org_module, nn.Linear):
                        module.ste0.V = 64
                    elif self.A_quant_method == "TwoRange" and isinstance(bit, int): 
                        assert not self.smooth_flag #TwoRange不支持同时开SmoothQunat
                        max_m = bit - 1
                        min_m = 1
                        if module.ste0.outputmaxminhook.output_min is not None:
                            output_min = float(module.ste0.outputmaxminhook.output_min)
                            output_max = float(module.ste0.outputmaxminhook.output_max)
                        abs_max = max(abs(output_min), abs(output_max))
                        hist = module.ste0.histhook.hist
                        hist_sum = hist.sum()
                        for m in range(max_m,min_m-1,-1):
                            tmp_min = max(0, round(255.0 / (output_max - output_min) * (max(output_min, -abs_max / 2.0**m) - output_min)))
                            tmp_max = min(255, round(255.0 / (output_max - output_min) * (min(output_max, abs_max / 2.0**m) - output_min)))
                            max(output_min, -abs_max / 2.0**m)
                            if hist[tmp_min:tmp_max].sum() / hist_sum > 0.95:
                                module.ste0.m = m
                                break
                    out_quant = module.org_module(module.ste0(inp[0])).to(torch.float32)
                    out = out.to(torch.float32)
                    if not hasattr(module, 'A_Loss'):
                        module.A_Loss = []
                    if j >= len(module.A_Loss):
                        if isinstance(bit, int) and self.smooth_flag:
                            if self.A_mp_method == "MSE":
                                module.A_Loss.append([F.mse_loss(out_quant, out).to(device='cpu')])
                            elif self.A_mp_method == "R2":
                                module.A_Loss.append([1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()])
                        else:
                            if self.A_mp_method == "MSE":
                                module.A_Loss.append(F.mse_loss(out_quant, out).to(device='cpu'))
                            elif self.A_mp_method == "R2":
                                module.A_Loss.append(1.0 - (out_quant - out).pow(2.0).sum() / (out.to(torch.float32) - out.mean()).pow(2.0).sum())
                    else:
                        if isinstance(bit, int) and self.smooth_flag:
                            if self.A_mp_method == "MSE":
                                module.A_Loss[j][0] = module.A_Loss[j][0].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                            elif self.A_mp_method == "R2":
                                module.A_Loss[j][0] = module.A_Loss[j][0].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu')
                        else:
                            if self.A_mp_method == "MSE":
                                module.A_Loss[j] = module.A_Loss[j].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                            elif self.A_mp_method == "R2":
                                module.A_Loss[j] = module.A_Loss[j].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu')
                    if isinstance(bit, int) and self.smooth_flag:
                        weight = module.org_module.weight.data.clone()
                        w_absmax = module.org_module.weight.data.abs().max(dim=0)[0]
                        a_absmax = torch.max(module.ste0.outputmaxminhook.output_smooth_min.abs(), module.ste0.outputmaxminhook.output_smooth_max.abs()).to(device)
                        for k in range(len(self.alpha)):
                            scales = (a_absmax.pow(self.alpha[j]) / w_absmax.pow(1-self.alpha[j])).clamp(min=torch.tensor(1e-5, dtype=dtype, device=device))
                            module.org_module.weight.data = weight * scales.view(1,-1)
                            module.ste0.k = ((1.0 / scales).half()).to(dtype)
                            module.ste0.outputmaxminhook.output_min = (module.ste0.outputmaxminhook.output_smooth_min.to(device) * (1.0 / scales)).min()
                            module.ste0.outputmaxminhook.output_max = (module.ste0.outputmaxminhook.output_smooth_max.to(device) * (1.0 / scales)).max()
                            out_quant = module.org_module(module.ste0(inp[0])).to(torch.float32)
                            out = out.to(torch.float32)
                            if (k+1) >= len(module.A_Loss[j]):
                                if self.A_mp_method == "MSE":
                                    module.A_Loss[j].append(F.mse_loss(out_quant, out).to(device='cpu'))
                                elif self.A_mp_method == "R2":
                                    module.A_Loss[j].append((1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu'))
                            else:
                                if self.A_mp_method == "MSE":
                                    module.A_Loss[j][k+1] = module.A_Loss[j][k+1].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                                elif self.A_mp_method == "R2":
                                    module.A_Loss[j][k+1].append(module.A_Loss[j][k+1].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu'))
                        module.org_module.weight.data = weight
                        module.ste0.k = None
                        module.ste0.outputmaxminhook.output_min = module.ste0.outputmaxminhook.output_smooth_min.min().to(device)
                        module.ste0.outputmaxminhook.output_max = module.ste0.outputmaxminhook.output_smooth_max.max().to(device)
                    module.ste0.n_bits = None
                    module.ste0.V = None
        elif isinstance(module, My_MatMul):
            device, dtype = inp[0].device, inp[0].dtype
            for j, bit in enumerate(self.A_MP_bit_list):
                if ((dtype == torch.float16 or dtype == torch.half) and 'fp16' in [bit]) or 'fp32' in [bit]:
                    if not hasattr(module, 'A_Loss'):
                        module.A_Loss = []
                    if j >= len(module.A_Loss):
                        if self.A_mp_method == "MSE":
                            module.A_Loss.append(torch.tensor(1e-10, device='cpu'))
                        elif self.A_mp_method == "R2":
                            module.A_Loss.append(torch.tensor(1.0, device='cpu'))
                    else:
                        if self.A_mp_method == "MSE":
                            module.A_Loss[j] = module.A_Loss[j] + torch.tensor(1e-10, device='cpu')
                        else:
                            module.A_Loss[j] = module.A_Loss[j] + torch.tensor(1.0, device='cpu')
                    if not hasattr(module, 'A_Loss_1'):
                        module.A_Loss_1 = []
                    if j >= len(module.A_Loss_1):
                        if self.A_mp_method == "MSE":
                            module.A_Loss_1.append(torch.tensor(1e-10, device='cpu'))
                        elif self.A_mp_method == "R2":
                            module.A_Loss_1.append(torch.tensor(1.0, device='cpu'))
                    else:
                        if self.A_mp_method == "MSE":
                            module.A_Loss_1[j] = module.A_Loss_1[j] + torch.tensor(1e-10, device='cpu')
                        elif self.A_mp_method == "R2":
                            module.A_Loss_1[j] = module.A_Loss_1[j] + torch.tensor(1.0, device='cpu')
                else:
                    module.ste0.n_bits = bit 
                    if self.A_quant_method == "VSQuant" and isinstance(bit, int) and isinstance(module.org_module, nn.Linear):
                        module.ste0.V = 64
                    elif self.A_quant_method == "TwoRange" and isinstance(bit, int): 
                        assert self.smooth_flag #TwoRange不支持同时开SmoothQunat
                        max_m = bit - 1
                        min_m = 1
                        if module.ste0.outputmaxminhook.output_min is not None:
                            output_min = float(module.ste0.outputmaxminhook.output_min)
                            output_max = float(module.ste0.outputmaxminhook.output_max)
                        abs_max = max(abs(output_min), abs(output_max))
                        hist = module.ste0.histhook.hist
                        hist_sum = hist.sum()
                        for m in range(max_m,min_m-1,-1):
                            tmp_min = max(0, round(255.0 / (output_max - output_min) * (max(output_min, -abs_max / 2.0**m) - output_min)))
                            tmp_max = min(255, round(255.0 / (output_max - output_min) * (min(output_max, abs_max / 2.0**m) - output_min)))
                            max(output_min, -abs_max / 2.0**m)
                            if hist[tmp_min:tmp_max].sum() / hist_sum > 0.95:
                                module.ste0.m = m
                                break
                    out_quant = torch.matmul(module.ste0(inp[0]), module.ste1(inp[1])).to(torch.float32)
                    out = out.to(torch.float32)
                    if not hasattr(module, 'A_Loss'):
                        module.A_Loss = []
                    if j >= len(module.A_Loss):
                        if self.A_mp_method == "MSE":
                            module.A_Loss.append(F.mse_loss(out_quant, out).to(device='cpu'))
                        elif self.A_mp_method == "R2":
                            module.A_Loss.append((1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu'))
                    else:
                        if self.A_mp_method == "MSE":
                            module.A_Loss[j] = module.A_Loss[j].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                        elif self.A_mp_method == "R2":
                            module.A_Loss[j] = module.A_Loss[j].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu')
                    module.ste0.n_bits = None
                    module.ste0.V = None
                    module.ste1.n_bits = bit 
                    if self.A_quant_method == "VSQuant" and isinstance(bit, int) and isinstance(module.org_module, nn.Linear):
                        module.ste1.V = 64
                    elif self.A_quant_method == "TwoRange" and isinstance(bit, int): 
                        assert self.smooth_flag #TwoRange不支持同时开SmoothQunat
                        max_m = bit - 1
                        min_m = 1
                        if module.ste1.outputmaxminhook.output_min is not None:
                            output_min = float(module.ste1.outputmaxminhook.output_min)
                            output_max = float(module.ste1.outputmaxminhook.output_max)
                        abs_max = max(abs(output_min), abs(output_max))
                        hist = module.ste1.histhook.hist
                        hist_sum = hist.sum()
                        for m in range(max_m,min_m-1,-1):
                            tmp_min = max(0, round(255.0 / (output_max - output_min) * (max(output_min, -abs_max / 2.0**m) - output_min)))
                            tmp_max = min(255, round(255.0 / (output_max - output_min) * (min(output_max, abs_max / 2.0**m) - output_min)))
                            max(output_min, -abs_max / 2.0**m)
                            if hist[tmp_min:tmp_max].sum() / hist_sum > 0.95:
                                module.ste1.m = m
                                break
                    out_quant = torch.matmul(module.ste1(inp[0]), module.ste1(inp[1])).to(torch.float32)
                    out = out.to(torch.float32)
                    if not hasattr(module, 'A_Loss_1'):
                        module.A_Loss_1 = []
                    if j >= len(module.A_Loss_1):
                        if self.A_mp_method == "MSE":
                            module.A_Loss_1.append(F.mse_loss(out_quant, out).to(device='cpu'))
                        elif self.A_mp_method == "R2":
                            module.A_Loss_1.append((1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu'))
                    else:
                        if self.A_mp_method == "MSE":
                            module.A_Loss_1[j] = module.A_Loss_1[j].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                        elif self.A_mp_method == "R2":
                            module.A_Loss_1[j] = module.A_Loss_1[j].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu')
                    module.ste1.n_bits = None
                    module.ste1.V = None
        torch.cuda.empty_cache()

@torch.no_grad()
def add_hook_A_mpqaunt(model: nn.Module, A_MP_bit_list, A_quant_method, A_mp_method, smooth_flag, alpha): #增加钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, (My_QuantModule, My_MatMul)):
            # print(name, child_module)
            child_module.A_mpqaunt_hook = A_mpqaunt_hook(A_MP_bit_list, A_quant_method, A_mp_method, smooth_flag, alpha)
            child_module.handle_A_mpqaunt = child_module.register_forward_hook(
                child_module.A_mpqaunt_hook)

@torch.no_grad()
def del_hook_A_mpqaunt(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, (My_QuantModule, My_MatMul)):
            # print(name, child_module)
            del child_module.A_mpqaunt_hook
            child_module.handle_A_mpqaunt.remove()

class W_mpqaunt_hook: #钩子函数
    def __init__(self, W_MP_bit_list, W_quant_method, W_mp_method, W_sys):
        super().__init__()
        self.W_MP_bit_list = W_MP_bit_list
        self.W_quant_method = W_quant_method
        self.W_mp_method = W_mp_method
        self.W_sys = W_sys

    @torch.no_grad()
    def __call__(self, module, inp, out):
        import torch.nn.functional as F
        if isinstance(module, My_QuantModule):
            device, dtype = module.org_module.weight.data.device, module.org_module.weight.data.dtype
            if not hasattr(self, 'W_Quant_tmp'):
                self.W_Quant_tmp = []
            for j, bit in enumerate(self.W_MP_bit_list):
                if ((dtype == torch.float16 or dtype == torch.half) and 'fp16' in [bit]) or 'fp32' in [bit]:
                    if not hasattr(module, 'W_Loss'):
                        module.W_Loss = []
                    if j >= len(module.W_Loss):
                        if self.W_mp_method == "MSE":
                            module.W_Loss.append(torch.tensor(1e-10, device='cpu'))
                        elif self.W_mp_method == "R2":
                            module.W_Loss.append(torch.tensor(1.0, device='cpu'))
                    else:
                        if self.W_mp_method == "MSE":
                            module.W_Loss[j] = module.W_Loss[j] + torch.tensor(1e-10, device='cpu')
                        elif self.W_mp_method == "R2":
                            module.W_Loss[j] = module.W_Loss[j] + torch.tensor(1.0, device='cpu')
                else:
                    weight = module.org_module.weight.data.clone()
                    if j >= len(self.W_Quant_tmp):
                        module.n_bits = bit 
                        if isinstance(bit, int):
                            module.W_sys = self.W_sys
                        if self.W_quant_method == 'Uniform':
                            module.W_UniformQuant()
                        elif self.W_quant_method == 'SQuant':
                            module.W_SQuant()
                        elif self.W_quant_method == 'VSQuant':
                            module.W_VSQuant()
                        elif self.W_quant_method == 'GPTQ':
                            module.W_GPTQ()
                        self.W_Quant_tmp.append(module.org_module.weight.data.clone().to(device='cpu'))
                    else:
                        module.org_module.weight.data = self.W_Quant_tmp[j].clone().to(device=device)
                    out_quant = module.org_module(module.ste0(inp[0])).to(torch.float32)
                    out = out.to(torch.float32)
                    if not hasattr(module, 'W_Loss'):
                        module.W_Loss = []
                    if j >= len(module.W_Loss):
                        if self.W_mp_method == "MSE":
                            module.W_Loss.append(F.mse_loss(out_quant, out).to(device='cpu'))
                        elif self.W_mp_method == "R2":
                            module.W_Loss.append((1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu'))
                    else:
                        if self.W_mp_method == "MSE":
                            module.W_Loss[j] = module.W_Loss[j].to(device='cpu') + F.mse_loss(out_quant, out).to(device='cpu')
                        elif self.W_mp_method == "R2":
                            module.W_Loss[j] = module.W_Loss[j].to(device='cpu') + (1.0 - (out_quant - out).pow(2.0).sum() / (out - out.mean()).pow(2.0).sum()).to(device='cpu')
                    module.org_module.weight.data = weight
                    module.n_bits = None
                    module.W_sys = True
        torch.cuda.empty_cache()

@torch.no_grad()
def add_hook_W_mpqaunt(model: nn.Module, W_MP_bit_list, W_quant_method, W_mp_method, W_sys): #增加钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule):
            # print(name, child_module)
            child_module.W_mpqaunt_hook = W_mpqaunt_hook(W_MP_bit_list, W_quant_method, W_mp_method, W_sys)
            child_module.handle_W_mpqaunt = child_module.register_forward_hook(
                child_module.W_mpqaunt_hook)

@torch.no_grad()
def del_hook_W_mpqaunt(model: torch.nn.Module): #删除钩子函数
    for name, child_module in model.named_modules():
        if isinstance(child_module, My_QuantModule):
            # print(name, child_module)
            # del child_module.W_mpqaunt_hook
            child_module.handle_W_mpqaunt.remove()
            
@torch.no_grad()
def get_cur_name(name):
    tmp_name = name.split('.')
    return tmp_name[-1]

@torch.no_grad()
def get_parent_module(model, name):
    tmp_name = name.split('.')
    str_name = "model"
    for j in range(len(tmp_name)-1):
        str_name = str_name + "._modules['" + tmp_name[j] + "']"
    return eval(str_name)

@torch.no_grad()
def get_related_module(model, name, related_name):
    tmp_name = name.split('.')
    str_name = "model"
    for j in range(len(tmp_name)-1):
        str_name = str_name + "._modules['" + tmp_name[j] + "']"
    str_name = str_name + "._modules['" + related_name + "']"
    return eval(str_name)
            
@torch.no_grad()
def WA_Quant(validate_calib, calib_loader, calib_num, calib_kwargs, 
             model, W_quant_method=None, A_quant_method=None, W_MP_bit_list=[4], A_MP_bit_list=[8], smooth_flag=False, W_sys=True, 
             W_mp_method="R2", A_mp_method="R2",
             W_MP_Th=0.8, A_MP_Th=0.8, W_plt_file=None, A_plt_file=None,
             related_list=[], search_iter=-1): #权重、数据量化
    from tqdm import tqdm
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    if A_quant_method is not None: #数据量化信息统计
        alpha = [0.5, 0.6, 0.7] #SmoothQuant alpha取值范围
        if search_iter <= 0:
            add_hook_A_mpqaunt(model, A_MP_bit_list, A_quant_method, A_mp_method, smooth_flag, alpha)
            print("validate_calib A MP Quant...")
            validate_calib(calib_output_flag=False, calib_loader=calib_loader,
                        calib_num=calib_num, model=model, **calib_kwargs)
            del_hook_A_mpqaunt(model)
        A_QuantModule_num, A_QuantModule_name, A_QuantModule_list = A_get_QuantModule_num(
            model)
        if search_iter > 0:
            for i in tqdm(range(A_QuantModule_num), 'A_Quant'):
                if isinstance(A_QuantModule_list[i], My_QuantModule):
                    A_QuantModule_list[i].ste0.n_bits = None
                    A_QuantModule_list[i].ste0.V = None
                else:
                    A_QuantModule_list[i].ste0.n_bits = None
                    A_QuantModule_list[i].ste1.n_bits = None
                    A_QuantModule_list[i].ste0.V = None
                    A_QuantModule_list[i].ste1.V = None
        A_org_sen_result = [0 for i in range(A_QuantModule_num)]
        A_sen_result = [0 for i in range(A_QuantModule_num)]
        A_sen_Th = [0 for i in range(A_QuantModule_num)]
        A_n_bits_buf = [0 for i in range(A_QuantModule_num)]
        alpha_buf = [0 for i in range(A_QuantModule_num)]
        for i in tqdm(range(A_QuantModule_num), 'A_Quant'):
            if isinstance(A_QuantModule_list[i], My_QuantModule):
                cur_device = next(A_QuantModule_list[i].parameters()).device
                if cur_device != device:
                    A_QuantModule_list[i].to(device)
                Th = torch.tensor(A_MP_Th)
                A_sen_Th[i] = Th
                for j, bit in enumerate(A_MP_bit_list):
                    if 0 == j:
                        if isinstance(A_QuantModule_list[i].A_Loss[j], list):
                            A_org_sen_result[i] = A_QuantModule_list[i].A_Loss[j][0] / calib_num
                        else:
                            A_org_sen_result[i] = A_QuantModule_list[i].A_Loss[j] / calib_num
                    if isinstance(A_QuantModule_list[i].A_Loss[j], list):
                        break_flag = False
                        for k in range(len(alpha)+1):
                            if ((A_QuantModule_list[i].A_Loss[j][k] / calib_num) < Th and A_mp_method == "MSE") \
                                or ((A_QuantModule_list[i].A_Loss[j][k] / calib_num) > Th and A_mp_method == "R2") \
                                or (k == (len(alpha) - 1) and bit in [A_MP_bit_list[-1]]):           
                                A_sen_result[i] = A_QuantModule_list[i].A_Loss[j][k] / calib_num
                                A_n_bits_buf[i] = bit
                                if k > 0:
                                    alpha_buf[i] = alpha[k-1]
                                break_flag = True
                                break   
                        if break_flag:
                            break                  
                    else:
                        if (A_QuantModule_list[i].A_Loss[j] / calib_num) < Th and A_mp_method == "MSE" \
                            or (A_QuantModule_list[i].A_Loss[j] / calib_num) > Th and A_mp_method == "R2" \
                            or bit in [A_MP_bit_list[-1]]: 
                            A_sen_result[i] = A_QuantModule_list[i].A_Loss[j] / calib_num
                            A_n_bits_buf[i] = bit
                            break
                if cur_device != device:
                    A_QuantModule_list[i].to(cur_device)
                torch.cuda.empty_cache()
            else:
                Th = torch.tensor(A_MP_Th)
                A_sen_Th[i] = Th   
                for j, bit in enumerate(A_MP_bit_list):
                    if A_mp_method == "MSE":
                        A_Loss = torch.max(A_QuantModule_list[i].A_Loss[j], A_QuantModule_list[i].A_Loss_1[j]) / calib_num
                    elif A_mp_method == "R2":
                        A_Loss = torch.min(A_QuantModule_list[i].A_Loss[j], A_QuantModule_list[i].A_Loss_1[j]) / calib_num
                    if 0 == j:
                        A_org_sen_result[i] = A_Loss
                    if A_Loss < Th  and A_mp_method == "MSE" \
                        or A_Loss > Th  and A_mp_method == "R2" \
                        or bit in [A_MP_bit_list[-1]]: 
                        A_sen_result[i] = A_Loss
                        A_n_bits_buf[i] = bit
                        break          
        for i in range(A_QuantModule_num): #根据related_list，统一相关模块的数据量化信息
            if isinstance(A_QuantModule_list[i], My_QuantModule):
                cur_device = next(A_QuantModule_list[i].parameters()).device
                cur_name = get_cur_name(A_QuantModule_name[i])
                parent_module = get_parent_module(model, A_QuantModule_name[i])
                related_flag = False
                for m, related in enumerate(related_list):
                    if cur_name in related:
                        related_flag = True
                        related_tmp = related
                        for n, r in enumerate(related):
                            if not hasattr(parent_module, r):
                                related_flag = False
                if related_flag:
                    fp32_flag = False
                    for n, r in enumerate(related_tmp):
                        j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                        if 'fp32' in [A_n_bits_buf[j]]:
                            fp32_flag = True
                            break
                    fp16_flag = False
                    if not fp32_flag:
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            if 'fp16' in [A_n_bits_buf[j]]:
                                fp16_flag = True
                                break
                    else:
                        k = A_MP_bit_list.index('fp32')
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            A_sen_result[j] = A_QuantModule_list[j].A_Loss[k] / calib_num
                            A_n_bits_buf[j] = 'fp32'
                    fp8_flag = False
                    if not fp32_flag and not fp16_flag:
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            if 'fp8' in [A_n_bits_buf[j]]:
                                fp8_flag = True
                                break
                    elif fp16_flag:
                        k = A_MP_bit_list.index('fp16')
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            A_sen_result[j] = A_QuantModule_list[j].A_Loss[k] / calib_num
                            A_n_bits_buf[j] = 'fp16'
                    max_alpha = 0
                    if not fp32_flag and not fp16_flag and not fp8_flag:
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            max_alpha = max(alpha_buf[j], max_alpha)
                    elif fp8_flag:
                        k = A_MP_bit_list.index('fp8')
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            A_sen_result[j] = A_QuantModule_list[j].A_Loss[k] / calib_num
                            A_n_bits_buf[j] = 'fp8'
                    if max_alpha != 0:
                        k = A_MP_bit_list.index(8)
                        m = alpha.index(max_alpha)+1
                        w_absmax = None
                        a_absmax = None
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            if w_absmax is None:
                                w_absmax = A_QuantModule_list[j].org_module.weight.data.abs().max(dim=0)[0]
                                a_absmax = torch.max(A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_min.abs(), A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_max.abs()).to(cur_device)
                            else:
                                w_absmax = torch.maximum(w_absmax, A_QuantModule_list[j].org_module.weight.data.abs().max(dim=0)[0])
                                a_absmax = torch.maximum(a_absmax, torch.max(A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_min.abs(), A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_max.abs()).to(cur_device))
                        for n, r in enumerate(related_tmp):
                            j = A_QuantModule_list.index(get_related_module(model, A_QuantModule_name[i], r))
                            A_sen_result[j] = A_QuantModule_list[j].A_Loss[k][m] / calib_num
                            A_n_bits_buf[j] = 8
                            alpha_buf[j] = max_alpha
                            if A_QuantModule_list[j].ste0.k is None:
                                scales = (a_absmax.pow(max_alpha) / w_absmax.pow(1-max_alpha)).clamp(min=torch.tensor(1e-5, dtype=dtype, device=cur_device))
                                A_QuantModule_list[j].org_module.weight.data = A_QuantModule_list[j].org_module.weight.data * scales.view(1,-1)
                                A_QuantModule_list[j].ste0.k = ((1.0 / scales).half()).to(dtype)
                                A_QuantModule_list[j].ste0.outputmaxminhook.output_min = (A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_min.to(cur_device) * (1.0 / scales)).min()
                                A_QuantModule_list[j].ste0.outputmaxminhook.output_max = (A_QuantModule_list[j].ste0.outputmaxminhook.output_smooth_max.to(cur_device) * (1.0 / scales)).max()        
        # A plot
        if A_plt_file is not None:
            import matplotlib.pyplot as plt
            marker_list = ['o', 's', 'd', '^', 'v', 'p']
            plt.style.use('ggplot')
            fig = plt.figure()
            # if A_mp_method == "MSE":
            plt.semilogy(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_org_sen_result],
                        linestyle='dotted', marker=marker_list[0], label='Org')
            plt.semilogy(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_sen_result],
                        linestyle='dotted', marker=marker_list[1], label='Mixed')
            plt.semilogy(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_sen_Th],
                        linestyle='dotted', marker=marker_list[2], label='Th')
            # elif A_mp_method == "R2":
            #     plt.plot(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_org_sen_result],
            #                 linestyle='dotted', marker=marker_list[0], label='Org')
            #     plt.plot(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_sen_result],
            #                 linestyle='dotted', marker=marker_list[1], label='Mixed')
            #     plt.plot(A_QuantModule_name, [x.detach().cpu().numpy() for x in A_sen_Th],
            #                 linestyle='dotted', marker=marker_list[2], label='Th')
            fig.autofmt_xdate()
            plt.grid(True)
            plt.legend()
            plt.title('A Sensitivity')
            plt.savefig(A_plt_file)
        
    if W_quant_method is not None: #权重量化信息统计
        if search_iter <= 0:
            add_hook_W_mpqaunt(model, W_MP_bit_list, W_quant_method, W_mp_method, W_sys)
            print("validate_calib W MP Quant...")
            validate_calib(calib_output_flag=False, calib_loader=calib_loader,
                        calib_num=calib_num, model=model, **calib_kwargs)
            del_hook_W_mpqaunt(model)
        W_QuantModule_num, W_QuantModule_name, W_QuantModule_list = W_get_QuantModule_num(
            model) 
        if search_iter > 0:
            for i in range(W_QuantModule_num):
                cur_device = next(W_QuantModule_list[i].parameters()).device
                W_QuantModule_list[i].n_bits = None
                W_QuantModule_list[i].org_module.weight.data = W_QuantModule_list[i].org_weight.clone().to(device=cur_device)
        W_org_sen_result = [0 for i in range(W_QuantModule_num)]
        W_sen_result = [0 for i in range(W_QuantModule_num)]       
        W_sen_Th = [0 for i in range(W_QuantModule_num)]    
        W_n_bits_buf = [0 for i in range(W_QuantModule_num)] 
        for i in range(W_QuantModule_num):
            if isinstance(W_QuantModule_list[i], My_QuantModule):
                cur_device = next(W_QuantModule_list[i].parameters()).device
                Th = torch.tensor(W_MP_Th)
                W_sen_Th[i] = Th
                for j, bit in enumerate(W_MP_bit_list):
                    if 0 == j:
                        W_org_sen_result[i] = W_QuantModule_list[i].W_Loss[j] / calib_num
                    if (W_QuantModule_list[i].W_Loss[j] / calib_num) < Th and W_mp_method == "MSE" \
                        or (W_QuantModule_list[i].W_Loss[j] / calib_num) > Th and W_mp_method == "R2" \
                        or bit in [W_MP_bit_list[-1]]: 
                        W_sen_result[i] = W_QuantModule_list[i].W_Loss[j] / calib_num
                        W_n_bits_buf[i] = bit
                        break
        # W plot
        if W_plt_file is not None:
            import matplotlib.pyplot as plt
            marker_list = ['o', 's', 'd', '^', 'v', 'p']
            plt.style.use('ggplot')
            fig = plt.figure()
            # if W_mp_method == "MSE":
            plt.semilogy(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_org_sen_result],
                        linestyle='dotted', marker=marker_list[0], label='Org')
            plt.semilogy(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_sen_result],
                        linestyle='dotted', marker=marker_list[1], label='Mixed')
            plt.semilogy(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_sen_Th],
                        linestyle='dotted', marker=marker_list[2], label='Th')
            # elif W_mp_method == "R2":
            #     plt.plot(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_org_sen_result],
            #                 linestyle='dotted', marker=marker_list[0], label='Org')
            #     plt.plot(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_sen_result],
            #                 linestyle='dotted', marker=marker_list[1], label='Mixed')
            #     plt.plot(W_QuantModule_name, [x.detach().cpu().numpy() for x in W_sen_Th],
            #                 linestyle='dotted', marker=marker_list[2], label='Th')
            fig.autofmt_xdate()
            plt.grid(True)
            plt.legend()
            plt.title('W Sensitivity')
            plt.savefig(W_plt_file)
            
    # A MP        
    if A_quant_method is not None: #数据量化       
        for i in range(A_QuantModule_num):
            if isinstance(A_QuantModule_list[i], My_QuantModule):
                A_QuantModule_list[i].ste0.n_bits = A_n_bits_buf[i]
                if A_quant_method == "VSQuant" and isinstance(A_n_bits_buf[i], int):
                    A_QuantModule_list[i].ste0.V = 64
                if search_iter == -1:
                    del A_QuantModule_list[i].A_Loss
            else:
                A_QuantModule_list[i].ste0.n_bits = A_n_bits_buf[i]
                A_QuantModule_list[i].ste1.n_bits = A_n_bits_buf[i]
                if A_quant_method == "VSQuant" and isinstance(A_n_bits_buf[i], int):
                    A_QuantModule_list[i].ste0.V = 64
                    A_QuantModule_list[i].ste1.V = 64
                if search_iter == -1:
                    del A_QuantModule_list[i].A_Loss
                    del A_QuantModule_list[i].A_Loss_1
        # A Stat
        A_sen_result = torch.tensor(A_sen_result).abs().detach().cpu().numpy()
        A_counter_all = 0
        A_counter_8 = 0
        A_counter_fp8 = 0
        A_counter_fp16 = 0
        A_counter_fp32 = 0
        for i in range(A_QuantModule_num):
            if A_QuantModule_list[i].ste0.n_bits != 8 or alpha_buf[i] != 0:
                if isinstance(A_QuantModule_list[i], My_QuantModule):
                    print(A_QuantModule_name[i], ": bit=", A_QuantModule_list[i].ste0.n_bits, ", alpha=", alpha_buf[i], ", sen=", A_sen_result[i], ", w_num=", A_QuantModule_list[i].org_module.weight.data.numel())
                else:
                    print(A_QuantModule_name[i], ": bit=", A_QuantModule_list[i].ste0.n_bits, ", sen=", A_sen_result[i])
            if A_QuantModule_list[i].ste0.n_bits == 8:
                A_counter_8 = A_counter_8 + 1
            elif A_QuantModule_list[i].ste0.n_bits == 'fp8':
                A_counter_fp8 = A_counter_fp8 + 1
            elif A_QuantModule_list[i].ste0.n_bits == 'fp16':
                A_counter_fp16 = A_counter_fp16 + 1
            else:
                A_counter_fp32 = A_counter_fp32 + 1
            A_counter_all = A_counter_all + 1
        if A_counter_all > 0:
            if A_counter_8 > 0:
                print('A 8 bit: ', A_counter_8 / A_counter_all * 100.0, '%, (', A_counter_8, '/', A_counter_all, ')')
            if A_counter_fp8 > 0:
                print('A fp8  : ', A_counter_fp8 / A_counter_all * 100.0, '%, (', A_counter_fp8, '/', A_counter_all, ')')
            if A_counter_fp16 > 0:
                print('A fp16 : ', A_counter_fp16 / A_counter_all * 100.0, '%, (', A_counter_fp16, '/', A_counter_all, ')')
            if A_counter_fp32 > 0:
                print('A fp32 : ', A_counter_fp32 / A_counter_all * 100.0, '%, (', A_counter_fp32, '/', A_counter_all, ')')
    # W MP
    if W_quant_method is not None: #权重量化信息统计
        for i in range(W_QuantModule_num):
            if isinstance(W_QuantModule_list[i], My_QuantModule):
                cur_device = next(W_QuantModule_list[i].parameters()).device
                if cur_device != device:
                    W_QuantModule_list[i].to(device)
                W_QuantModule_list[i].n_bits = W_n_bits_buf[i]
                W_QuantModule_list[i].W_sys = W_sys
                j = W_MP_bit_list.index(W_n_bits_buf[i])
                if search_iter == 0:
                    W_QuantModule_list[i].org_weight = W_QuantModule_list[i].org_module.weight.data.clone().to(device='cpu')
                if W_quant_method == 'Uniform':
                    W_QuantModule_list[i].org_module.weight.data = W_QuantModule_list[i].W_mpqaunt_hook.W_Quant_tmp[j].clone().to(device=device)
                elif W_quant_method == 'SQuant':
                    W_QuantModule_list[i].org_module.weight.data = W_QuantModule_list[i].W_mpqaunt_hook.W_Quant_tmp[j].clone().to(device=device)
                elif W_quant_method == 'VSQuant':
                    W_QuantModule_list[i].org_module.weight.data = W_QuantModule_list[i].W_mpqaunt_hook.W_Quant_tmp[j].clone().to(device=device)
                elif W_quant_method == 'GPTQ':
                    W_QuantModule_list[i].org_module.weight.data = W_QuantModule_list[i].W_mpqaunt_hook.W_Quant_tmp[j].clone().to(device=device)
                    if search_iter == -1:
                        del W_QuantModule_list[i].delta
                        if W_sys is False:
                            del W_QuantModule_list[i].zero_point
                        del W_QuantModule_list[i].gptq_hessian_hook.H
                        del W_QuantModule_list[i].Hinv
                        del W_QuantModule_list[i].perm
                if search_iter == -1:
                    del W_QuantModule_list[i].W_Loss
                    del W_QuantModule_list[i].W_mpqaunt_hook
                if cur_device != device:
                    W_QuantModule_list[i].to(cur_device)
                torch.cuda.empty_cache()
        # W Stat
        W_sen_result = torch.tensor(W_sen_result).abs().detach().cpu().numpy()
        counter_all = 0
        counter_2 = 0
        counter_4 = 0
        counter_8 = 0
        counter_fp8 = 0
        counter_fp16 = 0
        counter_fp32 = 0
        fp32_size = 0
        mp_size = 0
        for i in range(W_QuantModule_num):
            if W_QuantModule_list[i].n_bits != 4:
                print(W_QuantModule_name[i], ": bit=", W_QuantModule_list[i].n_bits, ", sen=", W_sen_result[i], ", w_num=", W_QuantModule_list[i].org_module.weight.data.numel())
            if W_QuantModule_list[i].n_bits == 2:
                counter_2 = counter_2 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 2.0 / 1024.0 / 1024.0 / 8.0
            elif W_QuantModule_list[i].n_bits == 4:
                counter_4 = counter_4 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 4.0 / 1024.0 / 1024.0 / 8.0
            elif W_QuantModule_list[i].n_bits == 8:
                counter_8 = counter_8 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 8.0 / 1024.0 / 1024.0 / 8.0
            elif W_QuantModule_list[i].n_bits == 'fp8':
                counter_fp8 = counter_fp8 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 8.0 / 1024.0 / 1024.0 / 8.0
            elif W_QuantModule_list[i].n_bits == 'fp16':
                counter_fp16 = counter_fp16 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 16.0 / 1024.0 / 1024.0 / 8.0
            else:
                counter_fp32 = counter_fp32 + 1
                mp_size = mp_size + W_QuantModule_list[i].org_module.weight.data.numel() * 32.0 / 1024.0 / 1024.0 / 8.0
            counter_all = counter_all + 1
            fp32_size = fp32_size + W_QuantModule_list[i].org_module.weight.data.numel() * 32.0 / 1024.0 / 1024.0 / 8.0
        if counter_all > 0:
            if counter_2 > 0:
                print('W 2 bit: ', counter_2 / counter_all * 100.0, '%, (', counter_2, '/', counter_all, ')')
            if counter_4 > 0:
                print('W 4 bit: ', counter_4 / counter_all * 100.0, '%, (', counter_4, '/', counter_all, ')')
            if counter_8 > 0:
                print('W 8 bit: ', counter_8 / counter_all * 100.0, '%, (', counter_8, '/', counter_all, ')')
            if counter_fp8 > 0:
                print('W fp8  : ', counter_fp8 / counter_all * 100.0, '%, (', counter_fp8, '/', counter_all, ')')
            if counter_fp16 > 0:
                print('W fp16 : ', counter_fp16 / counter_all * 100.0, '%, (', counter_fp16, '/', counter_all, ')')
            if counter_fp32 > 0:
                print('W fp32 : ', counter_fp32 / counter_all * 100.0, '%, (', counter_fp32, '/', counter_all, ')')
            print('W fp32  size(MB): ', fp32_size)
            print('W mixed size(MB): ', mp_size)
                
@torch.no_grad()
def My_Quant(validate_calib=None, 
                calib_loader=None,
                calib_num=None, 
                model=None, 
                replace_op_model=None,
                calib_kwargs={}, 
                W_quant_method=None, #None, 'Uniform', 'GPTQ'
                A_quant_method=None, #None, 'Uniform', 'TwoRange'
                W_MP_bit_list=[4],
                A_MP_bit_list=[8],
                smooth_flag=False,
                W_sys=True,
                W_mp_method="R2",
                A_mp_method="R2",
                W_MP_Th=0.8,
                A_MP_Th=0.8,
                W_plt_file=None,
                A_plt_file=None,
                related_list=[],
                org_replace_op_list=[],
                search_iter=-1
                ):
    if W_quant_method == 'None':
        W_quant_method = None
    if A_quant_method == 'None':
        A_quant_method = None
    if search_iter <= 0:
        org_op_list = []
        replace_op_list = []
        for val in org_replace_op_list:
            org_op_list.append(val[0])
            replace_op_list.append(val[1])
        if replace_op_model is None:
            replace_quant_op(model, tuple(org_op_list), tuple(replace_op_list))
        else:
            replace_quant_op(replace_op_model, tuple(org_op_list), tuple(replace_op_list))
        if A_quant_method is not None:
            add_hook_max_min_model(model)
            if smooth_flag:
                add_hook_smooth(model)
        if W_quant_method == 'GPTQ':
            add_hook_gptq_hessian(model)
        if A_quant_method is not None or (W_quant_method is not None and W_quant_method == 'GPTQ'):
            print("validate_calib MinMax/SmoothQuant or GPTQ Info...")
            validate_calib(calib_output_flag=False, calib_loader=calib_loader,
                        calib_num=calib_num, model=model, **calib_kwargs)
        if A_quant_method is not None:
            del_hook_max_min_model(model)
            if smooth_flag:
                del_hook_smooth(model)
        if W_quant_method == 'GPTQ':
            del_hook_gptq_hessian(model)

        if A_quant_method == 'TwoRange':    
            add_hook_hist(model)
            print("validate_calib Hist Info...")
            validate_calib(calib_output_flag=False, calib_loader=calib_loader,
                        calib_num=calib_num, model=model, **calib_kwargs)
            del_hook_hist(model)
        
    if W_quant_method is not None or A_quant_method is not None:
        WA_Quant(validate_calib, calib_loader, calib_num, calib_kwargs,
                 model, W_quant_method, A_quant_method, W_MP_bit_list, A_MP_bit_list, smooth_flag, W_sys, 
                 W_mp_method, A_mp_method,
                 W_MP_Th, A_MP_Th, W_plt_file, A_plt_file,
                 related_list, search_iter)
    
       
