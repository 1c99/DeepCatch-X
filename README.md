# DCX Medical Imaging Analysis - Code Duplication Analysis

## Extreme Duplication:
- **model.py**: 100% identical across 6 modules (22KB each)
- **base_options.py**: 99.9% identical (only 1 module has a single parameter difference)
- **utils.py**: Functionally identical (only line ending differences)

## High Duplication in Inference Scripts:
- Same preprocessing functions copy-pasted
- Same DICOM reading logic
- Same normalization code
- Main differences are just parameters: image size (512 vs 2048), thresholds (-1015 vs -950), output ranges

## Key Differences Are Minimal:
- Image resolution: 512x512 (heart/airway) vs 2048x2048 (lung/covid/vessel/bone)
- Some modules need lung masks as input (covid/vessel)
- Volumetry modules add a regression step
- Different thresholds and normalization ranges

## This could be refactored into:
1. A single shared base module with common utilities
2. Configuration files for each anatomy/task
3. One unified inference script with parameters
4. Shared model architecture code

The current structure has ~80% redundant code that could be consolidated into a cleaner, more maintainable architecture.