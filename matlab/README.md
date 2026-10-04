# MATLAB Dataset Generator

MATLAB source for the 100,000-sample 5G NR SNR estimation dataset used in the
paper. Requires MATLAB R2024a or later with the 5G Toolbox, Communications
Toolbox, Signal Processing Toolbox, and Statistics and Machine Learning Toolbox.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.m` | Generates the 100K main dataset (features + metadata CSVs). Runtime: ~2 hours on a standard laptop. |
| `generate_ood.m` | Regenerates only the 5K out-of-distribution set with shifted impairment severity. Runtime: ~3 minutes. |

## Running

```matlab
cd matlab
generate_dataset    % produces both main CSVs
generate_ood        % produces both OOD CSVs
