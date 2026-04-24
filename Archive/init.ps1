#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .\init.ps1 >>

# Create & activate virtual environment
python3.10 -m venv venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# -----------------------------
# Install core requirements
# -----------------------------
pip install -r requirements.txt

pip install onnxsim==0.4.36 --no-deps --force-reinstall

# -----------------------------
# Install Super-Gradients separately (NO deps)
# -----------------------------
pip install super-gradients==3.1.0 --no-deps

# -----------------------------
# DONE
# -----------------------------
Write-Host "✅ Setup Complete! Requirements + YOLO-NAS ready."