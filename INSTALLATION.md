# Installation Guide for SAKSHAM

## Step 1: Verify Python Installation

```bash
python --version
```
Should show Python 3.12.x

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# You should see (.venv) in your prompt
```

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Step 4: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0+ (with CUDA support if available)
- pandas, numpy, pyarrow (data processing)
- rich (beautiful console output)
- opencv-python, mediapipe (computer vision)
- And other dependencies

**Note:** Installation may take 5-10 minutes depending on your internet connection.

## Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import rich; print('Rich: OK')"
```

## Step 6: Run Quick Validation (No Dependencies)

Before installing everything, you can run a quick check:

```bash
python quick_validate.py
```

This uses only Python standard library and will:
- Check file existence
- Calculate basic statistics
- Generate a simple report

## Step 7: Run Full Validation (After Installing Dependencies)

```bash
python validate_dataset.py
```

This will:
- ✓ Verify all parquet files exist
- ✓ Check for corrupted parquet files
- ✓ Detect duplicates
- ✓ Validate labels
- ✓ Generate comprehensive markdown report

Output: `reports/dataset_report.md`

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Make sure virtual environment is activated and requirements are installed.

### Issue: PyTorch installation fails
**Solution:** Install PyTorch separately first:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### Issue: "command not found: python"
**Solution:** Try `python3` instead of `python` on some systems.

### Issue: Virtual environment activation fails
**Solution:** 
- Windows: Try `.venv\Scripts\Activate.ps1` in PowerShell
- Or check execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Next Steps

After successful installation and validation:
1. Review `reports/dataset_report.md`
2. Address any issues found
3. Proceed with preprocessing phase

## System Requirements

- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.12+
- **RAM:** 8GB+ recommended
- **Storage:** 10GB+ free space
- **GPU:** CUDA-capable GPU recommended (optional)
