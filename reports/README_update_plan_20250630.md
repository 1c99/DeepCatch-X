# File Created: 2025-06-30 10:51:00

# DCX Refactoring Plan

## Proposed Refactored Structure

```
dcx_unified/
├── core/
│   ├── base_options.py      # Single shared file
│   ├── utils.py            # Single shared file
│   ├── model.py            # Single shared file
│   ├── pix2pixHD_model.py  # Base class with variants
│   └── data_loader.py      # Base class with config
├── configs/
│   ├── lung.yaml           # image_size: 2048, threshold: -1015
│   ├── heart.yaml          # image_size: 512, threshold: -1015
│   ├── covid.yaml          # image_size: 2048, threshold: -950, needs_lung_mask: true
│   └── ...
├── inference.py            # Single unified inference script
└── modules/                # Keep existing checkpoints and data
```

## Example Config File (lung.yaml)

```yaml
name: "lung"
checkpoint: "checkpoints/xray2lung.pth"
image_size: 2048
threshold: -1015
output_range: [-1100, -500]
calculate_area: true
use_connected_regions: true
n_regions: 2
```

## Unified Inference Script

The single `inference.py` will:
1. Load config based on module name
2. Use exact same preprocessing logic
3. Apply module-specific parameters
4. Produce identical outputs

## Benefits

- **Maintainability**: Fix a bug once, it's fixed everywhere
- **Scalability**: Add new features to all modules at once
- **Testing**: Much easier to maintain and test
- **Consistency**: Exact same results as current implementation

## Special Cases

The bone suppression diffusion model will remain separate as it uses a completely different architecture.

## Implementation Steps

1. Create `dcx_unified/core/` directory with shared modules
2. Extract common code from all modules
3. Create configuration files for each anatomy/task
4. Implement unified inference script
5. Test each module to ensure identical outputs
6. Create comprehensive test suite
7. Document the new architecture