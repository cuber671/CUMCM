#!/usr/bin/env bash
# 一次性模型预热：联网环境跑通一次即可，之后离线复用（HF_HUB_OFFLINE=1）。
# 幂等：已下载的步骤会直接跳过/秒过，可反复执行直到全部就绪。
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== 1/4 bert-base-uncased（权重 ~440MB；tokenizer 已在本地）=="
if python -c "import transformers" 2>/dev/null; then
  python - <<'EOF'
import os
from transformers import AutoModel, AutoTokenizer
# 本地目录优先（cache/hf/bert-base-uncased，pin revision，见 env.sh）；缺权重时回退在线源
src = "cache/hf/bert-base-uncased"
if not os.path.exists(src + "/model.safetensors"):
    src = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(src)
mod = AutoModel.from_pretrained(src)
print("  bert ok:", mod.config.model_type, "| vocab:", tok.vocab_size, "| source:", src)
EOF
else
  echo "  跳过：transformers 未安装"
fi

echo "== 2/4 wav2vec2 forced_align 权重（~360MB，走 TORCH_HOME）=="
# bundle 型号与方案 §2.2 对应；若换 pipeline tag，务必同步写入 manifest
if python -c "import torch, torchaudio" 2>/dev/null; then
  python - <<'EOF'
import torch, torchaudio
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
print("  wav2vec2 ok | torch", torch.__version__, "| torchaudio", torchaudio.__version__)
EOF
else
  echo "  跳过：torch/torchaudio 未安装（走 PyPI 通道 torch==2.11.0，见方案 §8）"
fi

echo "== 3/4 Py-Feat 全套子模型（人脸检测/关键点/AU，首次 featurize 触发）=="
if python -c "import feat" 2>/dev/null; then
  python - <<'EOF'
from feat import Detector
det = Detector(device="cpu")
print("  py-feat ok")
EOF
  # 权重默认落 ~/.py-feat，统一收口到 cache/（env.sh 内有软链接命令）
else
  echo "  跳过：py-feat 未安装"
fi

echo "== 4/4 opensmile：无需下载（eGeMAPSv02 配置打包在 wheel 内）=="
python -c "import opensmile; print('  opensmile ok')" 2>/dev/null \
  || echo "  跳过：opensmile 未安装"

echo ""
echo "全部步骤 ok 后：核对各模型 revision/版本写入 manifest，再在 scripts/env.sh 打开 HF_HUB_OFFLINE=1"
