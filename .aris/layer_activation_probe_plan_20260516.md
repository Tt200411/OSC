# Layer Activation Mechanism Probe

**Date**: 2026-05-16  
**Research pipeline stage**: implementation/deploy preparation  
**Question**: Are current bounded-activation gains partly caused by applying `tanh` to the final output layer, rather than by hidden-layer activation dynamics?

## Code Evidence

Current Informer outputs are already linear by default:

- `lee_ocil/models/model.py`: `self.projection = nn.Linear(d_model, c_out, bias=True)`.
- The forward path applies the projection after the decoder and returns the prediction window.
- No `torch.tanh` or bounded function is applied to the prediction head unless the new diagnostic `--output_activation tanh` flag is explicitly used.

Therefore, existing `tanh`/`softsign` improvements cannot be explained by accidental output-range compression. They are effects of encoder/decoder feed-forward block activations.

## New Diagnostic Controls

The training entry point now accepts:

- `--encoder_activation`: activation used in encoder FFN blocks.
- `--decoder_activation`: activation used in decoder FFN blocks.
- `--output_activation`: `linear` by default, optional `tanh` counterfactual.

Backward compatibility is preserved:

- If `encoder_activation` and `decoder_activation` are omitted, both use `--activation`.
- If `output_activation` is omitted, the output layer remains linear.
- No parameter-count, attention, embedding, width, depth, or training-protocol change is introduced.

## Experiment Matrix

Script: `lee_ocil/scripts/run_phase1_layer_activation_probe.sh`

Default target cells:

- `Solar1`, `pred_len=96`
- `Solar5`, `pred_len=96`
- `ETTh1`, `pred_len=24`

Default seeds:

- `2024`, `2025`, `2026`

Compared configurations:

- all GELU hidden FFN, output linear
- all tanh hidden FFN, output linear
- encoder tanh / decoder GELU, output linear
- encoder GELU / decoder tanh, output linear
- all tanh hidden FFN, output tanh
- all `tanh_sin` with `a=0.01`, output linear

## Expected Interpretation

- If `all tanh, output tanh` is worse than `all tanh, output linear`, then output saturation is a harmful counterfactual and current gains are not range-matching artifacts.
- If encoder-only or decoder-only tanh recovers most of all-tanh improvement, the mechanism localizes to that side of the Informer FFN stack.
- If neither partial setting recovers the all-tanh gain, the effect likely depends on both encoder and decoder hidden transformations.
- If `tanh_sin a=0.01` helps only in factor-conditioned bins, the oscillation claim should remain conditional rather than global.

## Current Deployment Status

Local implementation and static checks pass. SSH from this environment to `10.20.12.248` and `10.21.53.62` timed out at TCP port 22 on 2026-05-16, so remote launch is blocked by network reachability rather than credentials or code.
