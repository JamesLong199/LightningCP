# BFloat16 Tensor Saving Fix

## Problem

NumPy doesn't natively support saving tensors with the BFloat16 data type. When running the MEMO model with BFloat16 precision, attempting to save tensors directly using `np.save()` results in the following error:

```
TypeError: Got unsupported ScalarType BFloat16
```

## Solution

We've implemented several fixes to address this issue:

1. Added `.float()` conversion before `.numpy()` for all tensor saving operations:
   ```python
   # Before:
   np.save(filepath, tensor.detach().cpu().numpy())
   
   # After:
   np.save(filepath, tensor.detach().cpu().float().numpy())
   ```

2. Created a utility function `safe_save_tensor()` in `memo/utils/tensor_utils.py` to handle any tensor saving operations safely:
   ```python
   from memo.utils.tensor_utils import safe_save_tensor
   
   # Use this instead of np.save directly
   safe_save_tensor(tensor, filepath)
   ```

## Files Modified

- `memo/pipelines/video_pipeline.py`
- `memo/pipelines/video_pipeline_multi_gpu.py`
- `memo/models/masking_utils.py`
- Added new file: `memo/utils/tensor_utils.py`

## Future Development

When working with BFloat16 or other non-standard data types and you need to save tensors:

1. Always convert tensors to standard floating-point formats (float32) before saving to NumPy arrays
2. Use `safe_save_tensor()` from the utils module for consistent behavior
3. Be particularly careful when using BFloat16 with debugging operations that save intermediates

If you encounter similar errors with other tensor operations, check if you need to convert the tensor to a standard data type first. 