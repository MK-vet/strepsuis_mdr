# Installation Guide

This guide will help you install Python and all required dependencies for the StrepSuis-MDR module.

## Prerequisites

- **Python 3.8 or higher** (Python 3.9, 3.10, 3.11, or 3.12 recommended)
- **pip** (usually comes with Python)
- **Windows PowerShell** or **Command Prompt**

## Step 1: Install Python

If Python is not installed on your system:

1. Download Python from: https://www.python.org/downloads/
2. **IMPORTANT**: During installation, check the box **"Add Python to PATH"**
3. Complete the installation
4. **Restart your terminal/PowerShell** after installation

### Verify Python Installation

Open PowerShell or Command Prompt and run:

```powershell
python --version
```

You should see something like: `Python 3.11.x`

If you see an error, Python is not in your PATH. Reinstall Python and make sure to check "Add Python to PATH".

## Step 2: Install Dependencies

### Option A: Using PowerShell Script (Recommended)

1. Open PowerShell in the `strepsuis-mdr` directory
2. Run:

```powershell
.\install_dependencies.ps1
```

If you get an execution policy error, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_dependencies.ps1
```

### Option B: Using Batch Script

1. Open Command Prompt in the `strepsuis-mdr` directory
2. Run:

```cmd
install_dependencies.bat
```

### Option C: Manual Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate the virtual environment:

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

3. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Install the package in editable mode:

```powershell
pip install -e .
```

## Step 3: Verify Installation

After installation, verify that all packages are installed:

```powershell
python -c "import pandas, numpy, scipy, networkx, plotly, statsmodels, mlxtend; print('All packages installed successfully!')"
```

## Troubleshooting

### Python Not Found

- Make sure Python is installed and added to PATH
- Restart your terminal after installing Python
- Try using `python3` instead of `python`
- Check if Python is in: `C:\Users\YourName\AppData\Local\Programs\Python\`

### pip Not Found

- Python should come with pip, but if not: `python -m ensurepip --upgrade`
- Or download get-pip.py from: https://pip.pypa.io/en/stable/installation/

### Virtual Environment Issues

- Delete the `.venv` folder and recreate it
- Make sure you're in the correct directory
- Try using `python -m venv .venv` instead of `venv .venv`

### Package Installation Errors

- Make sure you're in the virtual environment (you should see `(.venv)` in your prompt)
- Try upgrading pip: `python -m pip install --upgrade pip`
- Some packages may require Visual C++ Build Tools on Windows
- For networkx, scipy, numpy: These are large packages and may take time to install

### Import Errors After Installation

- Make sure the virtual environment is activated
- Verify packages are installed: `pip list`
- Reinstall problematic packages: `pip install --force-reinstall package_name`

## Required Packages

The following packages will be installed:

- **Core**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly, kaleido
- **Statistics**: statsmodels, scikit-learn
- **Network Analysis**: networkx
- **Association Rules**: mlxtend
- **Bioinformatics**: biopython
- **Excel Reports**: openpyxl, xlsxwriter
- **Utilities**: tqdm, joblib, jinja2

## Next Steps

After successful installation:

1. Activate the virtual environment (if not already active)
2. Run the analysis: `strepsuis-mdr analyze --help`
3. Or use Python directly: `python -m strepsuis_mdr.mdr_analysis_core`

## Support

If you encounter issues:

1. Check the error messages carefully
2. Verify Python version: `python --version` (must be 3.8+)
3. Check if virtual environment is activated
4. Try reinstalling: delete `.venv` and run installation script again


