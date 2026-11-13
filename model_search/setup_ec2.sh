#!/bin/bash
# One-time setup on the EC2 GPU box (Deep Learning Base OSS Nvidia Driver AMI, Ubuntu 22.04).
#   bash setup_ec2.sh            -> ~/venv with TF 2.14.1 + CUDA 11.8 libs, tflite-runtime, kaggle; Claude Code
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update -qq && sudo apt-get install -y -qq tmux unzip >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/venv ] || uv venv -q --seed --python 3.11 ~/venv
PY=~/venv/bin/python
uv pip install -q --python $PY "tensorflow==2.14.1" "numpy<2" pandas "pyarrow<19" kaggle "tflite-runtime==2.14.0"
# CUDA libs TF 2.14 lists under its and-cuda extra, minus tensorrt (no installable wheel; optional)
CUDA=$($PY - <<'EOF'
from importlib.metadata import requires
out = []
for r in requires('tensorflow'):
    if 'and-cuda' in r and not r.startswith('tensorrt'):          # e.g. "nvidia-cublas-cu11 (==11.11.3.6) ; extra == 'and-cuda'"
        out.append(r.split(';')[0].replace(' ', '').replace('(', '').replace(')', ''))
if not out:
    raise SystemExit('could not read the CUDA requirements from tensorflow metadata')
print(' '.join(out))
EOF
)
echo "CUDA libs: $CUDA"
uv pip install -q --python $PY $CUDA
LIBS=$($PY -c "import nvidia, os; b = os.path.dirname(nvidia.__path__[0]) + '/nvidia'; print(':'.join(sorted(os.path.join(b, d, 'lib') for d in os.listdir(b) if os.path.isdir(os.path.join(b, d, 'lib')))))")
cat > ~/env.sh <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
export LD_LIBRARY_PATH="$LIBS:\${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
source ~/venv/bin/activate
EOF
source ~/env.sh
python -c "import tensorflow as tf, tflite_runtime, pyarrow; g = tf.config.list_physical_devices('GPU'); print('TF', tf.__version__, 'GPUs', g); assert g, 'NO GPU VISIBLE'"
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash
echo "SETUP OK"
