import torch
import torch.nn as nn
import torch.nn.functional as F

from lee_oc import LeeOscillator


BASELINE_ACTIVATIONS = {"gelu", "relu"}
BOUNDED_SIGN_ACTIVATIONS = {"tanh", "softsign", "scaled_tanh"}
FUNCTIONAL_ACTIVATIONS = BASELINE_ACTIVATIONS | BOUNDED_SIGN_ACTIVATIONS
SIN_ACTIVATIONS = {"gelu_sin", "relu_sin", "tanh_sin"}


def activation_family(activation):
    if activation in BASELINE_ACTIVATIONS:
        return "baseline"
    if activation in BOUNDED_SIGN_ACTIVATIONS:
        return "bounded_sign"
    if activation in SIN_ACTIVATIONS:
        return "sin_perturb"
    if activation == "dynamic_gelu_sin":
        return "dynamic_sin_perturb"
    if activation in {"lee", "lee_oc", "lee-oc"}:
        return "lee_oc"
    return "unknown"


class FunctionalActivation(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in FUNCTIONAL_ACTIVATIONS:
            raise ValueError(f"Unsupported functional activation: {name}")
        self.name = name

    def forward(self, x):
        if self.name == "gelu":
            return F.gelu(x)
        if self.name == "relu":
            return F.relu(x)
        if self.name == "tanh":
            return torch.tanh(x)
        if self.name == "softsign":
            return F.softsign(x)
        if self.name == "scaled_tanh":
            return torch.tanh(2.0 * x)
        raise RuntimeError(f"Unsupported functional activation: {self.name}")


class SinPerturbedActivation(nn.Module):
    def __init__(self, base_activation, amplitude=0.0, frequency=1.0, phase=0.0):
        super().__init__()
        if base_activation not in FUNCTIONAL_ACTIVATIONS:
            raise ValueError(f"Unsupported sin base activation: {base_activation}")
        self.base = FunctionalActivation(base_activation)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)

    def _amplitude_for(self, x):
        return self.amplitude

    def forward(self, x):
        return self.base(x) + self._amplitude_for(x) * torch.sin(
            self.frequency * x + self.phase
        )


class DynamicGeluSinActivation(SinPerturbedActivation):
    def __init__(self, frequency=1.0, phase=0.0, default_amplitude=0.05):
        super().__init__(
            base_activation="gelu",
            amplitude=default_amplitude,
            frequency=frequency,
            phase=phase,
        )
        self.current_amplitude = None

    def set_dynamic_amplitude(self, amplitude):
        self.current_amplitude = amplitude

    def _amplitude_for(self, x):
        if self.current_amplitude is None:
            return self.amplitude
        amplitude = self.current_amplitude.to(device=x.device, dtype=x.dtype)
        while amplitude.dim() < x.dim():
            amplitude = amplitude.unsqueeze(-1)
        return amplitude


class LeeActivation(nn.Module):
    def __init__(self, lee_type=1):
        super().__init__()
        self.lee_type = int(lee_type)
        self.lee = LeeOscillator()

    def forward(self, x):
        return self.lee(x, oscillator_type=self.lee_type)


def build_activation(
    activation,
    perturb_amplitude=0.0,
    perturb_frequency=1.0,
    perturb_phase=0.0,
    lee_type=1,
    dynamic_default_amplitude=0.05,
):
    activation = activation.lower()
    if activation in FUNCTIONAL_ACTIVATIONS:
        return FunctionalActivation(activation)
    if activation == "gelu_sin":
        return SinPerturbedActivation(
            "gelu", perturb_amplitude, perturb_frequency, perturb_phase
        )
    if activation == "relu_sin":
        return SinPerturbedActivation(
            "relu", perturb_amplitude, perturb_frequency, perturb_phase
        )
    if activation == "tanh_sin":
        return SinPerturbedActivation(
            "tanh", perturb_amplitude, perturb_frequency, perturb_phase
        )
    if activation == "dynamic_gelu_sin":
        return DynamicGeluSinActivation(
            frequency=perturb_frequency,
            phase=perturb_phase,
            default_amplitude=dynamic_default_amplitude,
        )
    if activation in {"lee", "lee_oc", "lee-oc"}:
        return LeeActivation(lee_type=lee_type)
    raise ValueError(f"Unsupported activation: {activation}")
