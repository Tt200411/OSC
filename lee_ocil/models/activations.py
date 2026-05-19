import torch
import torch.nn as nn
import torch.nn.functional as F

from lee_oc import LeeOscillator


BASELINE_ACTIVATIONS = {"gelu", "relu", "swish", "silu"}
BOUNDED_SIGN_ACTIVATIONS = {"tanh", "softsign", "scaled_tanh"}
FUNCTIONAL_ACTIVATIONS = BASELINE_ACTIVATIONS | BOUNDED_SIGN_ACTIVATIONS
SIN_ACTIVATIONS = {"gelu_sin", "relu_sin", "tanh_sin"}
COS_ACTIVATIONS = {"gelu_cos", "relu_cos", "tanh_cos"}
RANDOM_OSC_ACTIVATIONS = {"tanh_rand"}


def activation_family(activation):
    if activation in BASELINE_ACTIVATIONS:
        return "baseline"
    if activation in BOUNDED_SIGN_ACTIVATIONS:
        return "bounded_sign"
    if activation in SIN_ACTIVATIONS:
        return "sin_perturb"
    if activation in COS_ACTIVATIONS:
        return "cos_perturb"
    if activation in RANDOM_OSC_ACTIVATIONS:
        return "random_osc_perturb"
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
        if self.name in {"swish", "silu"}:
            return F.silu(x)
        if self.name == "tanh":
            return torch.tanh(x)
        if self.name == "softsign":
            return F.softsign(x)
        if self.name == "scaled_tanh":
            return torch.tanh(2.0 * x)
        raise RuntimeError(f"Unsupported functional activation: {self.name}")


class OscillatoryPerturbedActivation(nn.Module):
    def __init__(
        self,
        base_activation,
        amplitude=0.0,
        frequency=1.0,
        phase=0.0,
        waveform="sin",
    ):
        super().__init__()
        if base_activation not in FUNCTIONAL_ACTIVATIONS:
            raise ValueError(f"Unsupported oscillatory base activation: {base_activation}")
        if waveform not in {"sin", "cos", "rand"}:
            raise ValueError(f"Unsupported oscillatory waveform: {waveform}")
        self.base = FunctionalActivation(base_activation)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        self.waveform = waveform

    def _amplitude_for(self, x):
        return self.amplitude

    def _oscillation(self, x):
        argument = self.frequency * x + self.phase
        if self.waveform == "sin":
            return torch.sin(argument)
        if self.waveform == "cos":
            return torch.cos(argument)
        # Fixed random-Fourier style perturbation. It is deterministic, so runs stay
        # reproducible while still testing a non-single-frequency oscillatory shape.
        return 0.5 * torch.sin(argument + 0.37) + 0.5 * torch.cos(
            1.61803398875 * argument - 1.17
        )

    def forward(self, x):
        return self.base(x) + self._amplitude_for(x) * self._oscillation(x)


SinPerturbedActivation = OscillatoryPerturbedActivation


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
        return OscillatoryPerturbedActivation(
            "gelu", perturb_amplitude, perturb_frequency, perturb_phase, "sin"
        )
    if activation == "relu_sin":
        return OscillatoryPerturbedActivation(
            "relu", perturb_amplitude, perturb_frequency, perturb_phase, "sin"
        )
    if activation == "tanh_sin":
        return OscillatoryPerturbedActivation(
            "tanh", perturb_amplitude, perturb_frequency, perturb_phase, "sin"
        )
    if activation == "gelu_cos":
        return OscillatoryPerturbedActivation(
            "gelu", perturb_amplitude, perturb_frequency, perturb_phase, "cos"
        )
    if activation == "relu_cos":
        return OscillatoryPerturbedActivation(
            "relu", perturb_amplitude, perturb_frequency, perturb_phase, "cos"
        )
    if activation == "tanh_cos":
        return OscillatoryPerturbedActivation(
            "tanh", perturb_amplitude, perturb_frequency, perturb_phase, "cos"
        )
    if activation == "tanh_rand":
        return OscillatoryPerturbedActivation(
            "tanh", perturb_amplitude, perturb_frequency, perturb_phase, "rand"
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
