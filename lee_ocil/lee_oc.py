import torch
import torch.nn as nn
import math

class LeeOscillator(nn.Module):
    def __init__(self):
        super(LeeOscillator, self).__init__()
        # 通用参数
        self.N = 100          # 时间步数
        self.e = 0.001        # 外部刺激大小
        self.k = 500          # K 值
        
        # 生成刺激范围张量 (-1 到 1)
        self.register_buffer('i_values', torch.arange(-1, 1, 0.001))
    
    def _tanh5(self, x):
        """通用tanh激活函数"""
        if not torch.is_tensor(x):
            return math.tanh(5 * x)
        return torch.tanh(5 * x)
    
    def _tanh(self, x):
        """通用tanh激活函数"""
        if not torch.is_tensor(x):
            return math.tanh(x)
        return torch.tanh(x)


    def type1(self, x):
        """第一类Lee振荡器"""
        # 参数设置
        a1, a2, a3, a4 = 0, 5.0, 5.0, 1.0
        b1, b2, b3, b4 = 0, -1.0, 1.0, 0
        xi_E, xi_I = 0, 0.0
        
        return self._run_oscillator(x, self._tanh5(a1), self._tanh5(a2), self._tanh5(a3), self._tanh5(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type2(self, x):
        """第二类Lee振荡器"""
        a1, a2, a3, a4 = 0.5, 0.55, 0.55, -0.5
        b1, b2, b3, b4 = 0.5, -0.55, -0.55, -0.5
        xi_E, xi_I = 0, 0.0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type3(self, x):
        """第三类Lee振荡器"""
        a1, a2, a3, a4 = -5, 5, 5, -5
        b1, b2, b3, b4 = 1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type4(self, x):
        """第四类Lee振荡器"""
        a1, a2, a3, a4 = 1, 1, 1, -1
        b1, b2, b3, b4 = -1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type5(self, x):
        """第五类Lee振荡器"""
        a1, a2, a3, a4 = 5, -5, -5, 5
        b1, b2, b3, b4 = -1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type6(self, x):
        """第六类Lee振荡器"""
        a1, a2, a3, a4 = -1, -1, -1, 1
        b1, b2, b3, b4 = 1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type7(self, x):
        """第七类Lee振荡器"""
        a1, a2, a3, a4 = 1, -1, -1, 1
        b1, b2, b3, b4 = -1, 1, 1, -1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def type8(self, x):
        """第八类Lee振荡器"""
        a1, a2, a3, a4 = -1, 1, 1, -1
        b1, b2, b3, b4 = 1, -1, -1, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)
    
    def type9(self, x):
        """第九类Lee振荡器"""
        a1, a2, a3, a4 = 1, -1, -1, -1  
        b1, b2, b3, b4 =  -1, 2, 2, -1 
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)
    
    def type10(self, x):
        """第十类Lee振荡器"""
        a1, a2, a3, a4 = 3, 3, 3, 2 
        b1, b2, b3, b4 =  0.45, -0.45, -0.45, 1
        xi_E, xi_I = 0, 0
        
        return self._run_oscillator(x, self._tanh(a1), self._tanh(a2), self._tanh(a3), self._tanh(a4), 
                                     b1, b2, b3, b4, xi_E, xi_I)

    def _run_oscillator(self, x, a1, a2, a3, a4, b1, b2, b3, b4, xi_E, xi_I):
        """运行Lee振荡器的核心函数 - vectorized PyTorch版本"""
        device = x.device
        dtype = x.dtype

        a1 = torch.as_tensor(a1, device=device, dtype=dtype)
        a2 = torch.as_tensor(a2, device=device, dtype=dtype)
        a3 = torch.as_tensor(a3, device=device, dtype=dtype)
        a4 = torch.as_tensor(a4, device=device, dtype=dtype)
        b1 = torch.as_tensor(b1, device=device, dtype=dtype)
        b2 = torch.as_tensor(b2, device=device, dtype=dtype)
        b3 = torch.as_tensor(b3, device=device, dtype=dtype)
        b4 = torch.as_tensor(b4, device=device, dtype=dtype)
        xi_E = torch.as_tensor(xi_E, device=device, dtype=dtype)
        xi_I = torch.as_tensor(xi_I, device=device, dtype=dtype)

        sim = x + self.e * torch.sign(x)
        exp_term = torch.exp(-self.k * sim * sim)
        omega = torch.tanh(sim)

        E = torch.full_like(x, 0.2)
        I = torch.zeros_like(x)
        lors = torch.full_like(x, 0.2)

        for _ in range(self.N - 1):
            next_E = torch.tanh(a1 * lors + a2 * E - a3 * I + a4 * sim - xi_E)
            next_I = torch.tanh(b1 * lors - b2 * E - b3 * I + b4 * sim - xi_I)
            lors = (next_E - next_I) * exp_term + omega
            E, I = next_E, next_I

        return lors

    def forward(self, x, oscillator_type=1):
        """前向传播函数，方便在nn.Module中使用"""
        oscillator_funcs = {
            1: self.type1,
            2: self.type2,
            3: self.type3,
            4: self.type4,
            5: self.type5,
            6: self.type6,
            7: self.type7,
            8: self.type8,
            9: self.type9,
            10: self.type10
        }
        return oscillator_funcs[oscillator_type](x)
