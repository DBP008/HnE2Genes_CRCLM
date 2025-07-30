# Spatial Attention Heatmap Visualization for CLAM Training

## Overview

This implementation adds **spatial** attention heatmap visualization to TensorBoard during CLAM_SB model training for binary classification tasks. The visualization shows how the model's attention evolves throughout training on the first validation sample, with patches positioned according to their actual coordinates in the WSI.

## What's Been Added

### 1. New Functions

#### `create_spatial_attention_heatmap()`
- Creates a **spatially-aware** heatmap using actual patch coordinates from the WSI
- Maps patches to their true positions in the slide
- Handles overlapping patches by averaging attention scores
- Downsamples for efficient visualization

#### `create_attention_heatmap()` (Legacy)
- Creates a simple grid-based arrangement when coordinates are unavailable
- Maintained for backward compatibility

#### `get_coords_from_slide_data()`
- Extracts patch coordinates from h5 files in the dataset
- Handles different path structures automatically
- Provides robust error handling

### 2. Enhanced `validate_clam()` Function

The validation function now:
- Captures both attention scores AND coordinates from the first validation sample
- Automatically detects coordinate availability
- Creates appropriate visualization based on available data
- Only activates for CLAM_SB models with binary classification

### 3. Intelligent TensorBoard Logging

**Two Types of Heatmaps:**
- `val/spatial_attention_heatmap`: When coordinates are available - shows true spatial distribution
- `val/grid_attention_heatmap`: When coordinates are unavailable - shows grid arrangement

## Key Features

- ✅ **True spatial representation**: Patches positioned by actual WSI coordinates
- ✅ **Automatic fallback**: Uses grid layout when coordinates unavailable  
- ✅ **Coordinate validation**: Ensures attention-coordinate count matching
- ✅ **Overlap handling**: Averages attention scores for overlapping regions
- ✅ **Memory efficient**: Proper matplotlib figure cleanup
- ✅ **Robust error handling**: Won't interrupt training if visualization fails

## Spatial Heatmap Details

### Coordinate System
- Uses actual patch coordinates from the WSI (x, y positions)
- Automatically calculates canvas size based on coordinate extent
- Applies downsampling for efficient visualization (default: 32x)

### Canvas Creation
- **Canvas size**: Based on WSI extent divided by downsample factor
- **Patch mapping**: Each patch mapped to its true position
- **Overlap resolution**: Multiple patches per pixel are averaged

### Visualization Parameters
- **Downsample factor**: 32x (configurable in function)
- **Patch size**: 256 pixels (configurable)
- **Colormap**: "Hot" (black=low attention, white/yellow=high attention)
- **Interpolation**: Bilinear for smooth appearance

## Usage

The visualization is automatically enabled when:
1. Using CLAM_SB model
2. Binary classification task (`n_classes == 2`)
3. TensorBoard writer is available
4. Validation data is present

### In TensorBoard:
- **Images tab** → `val/spatial_attention_heatmap` (when coordinates available)
- **Images tab** → `val/grid_attention_heatmap` (fallback mode)

## Technical Implementation

### Data Flow
1. **Training epoch** → Validation phase
2. **First validation sample** → Extract attention scores + coordinates
3. **H5 file lookup** → Load patch coordinates from dataset
4. **Spatial mapping** → Create canvas and map patches to positions
5. **TensorBoard logging** → Upload visualization

### Error Handling
- Missing coordinate files → Falls back to grid layout
- Mismatched attention/coordinate counts → Uses grid layout with warning
- H5 file errors → Graceful degradation with informative messages
- Visualization errors → Warning messages without training interruption

## Example Output

### Spatial Heatmap
Shows attention distributed across the actual WSI space:
- **X-axis**: WSI width (downsampled coordinates)
- **Y-axis**: WSI height (downsampled coordinates)  
- **Color intensity**: Attention strength at each position
- **Spatial context**: Maintains tissue architecture relationships

### Console Output
```
Creating spatial attention heatmap with 1024 patches and 1024 coordinates
```
or
```
Warning: Coordinate count (512) doesn't match attention count (1024), using grid layout
Creating grid attention heatmap with 1024 patches
```

## Benefits for Analysis

1. **Anatomical Context**: See which tissue regions the model focuses on
2. **Spatial Patterns**: Identify if attention follows tissue boundaries
3. **Training Evolution**: Watch how spatial attention patterns change over epochs
4. **Model Debugging**: Detect if model focuses on artifacts or irrelevant regions
5. **Biological Insight**: Understand model's decision-making in tissue context

This spatial approach provides much more meaningful visualization than simple grid arrangements, making it easier to understand and interpret the model's attention behavior in the context of actual tissue morphology.

## Configuration Options

You can customize the spatial heatmap by modifying parameters in `create_spatial_attention_heatmap()`:

```python
def create_spatial_attention_heatmap(attention_scores, coords, 
                                   patch_size=256,        # Original patch size
                                   downsample_factor=32): # Visualization downsample
```

- **patch_size**: Size of patches in pixels (should match training patch size)
- **downsample_factor**: How much to downsample for visualization (higher = smaller heatmap, faster)

## Troubleshooting

### Common Issues:

1. **"Could not load coordinates"**: H5 file path issue
   - Check that h5 files exist in expected location
   - Verify slide_id format matches file names

2. **"Coordinate count doesn't match"**: Feature/coordinate mismatch
   - Usually indicates different patch extraction parameters
   - Falls back to grid layout automatically

3. **Empty heatmap**: All attention scores are zero
   - Check if model is properly loaded
   - Verify attention mechanism is working

### Debug Information:
The implementation provides detailed console output to help diagnose issues:
- Coordinate loading success/failure
- Attention/coordinate count matching
- Heatmap type being created
