# 模型缓存统一收口：所有管线脚本入口先 source 本文件
# 约定：权重不入库（.gitignore 已覆盖 cache/），复现靠 revision pin + manifest
# bash/zsh 通用：bash 取 BASH_SOURCE，zsh source 时 $0 即本文件路径
_script="${BASH_SOURCE[0]:-$0}"
export CUMCM_ROOT="$(cd "$(dirname "$_script")/.." && pwd)"

# bert-base-uncased 等 HF 权重与 tokenizer（首跑自动落盘）
export HF_HOME="$CUMCM_ROOT/cache/hf"

# 网络受限环境实测（2026-09-23）：cdn-lfs.huggingface.co 不可达，hf CLI 卡死；
# 可用通道 = hf-mirror.com（/api 与 /resolve 均需跟随重定向，LFS 速度 ~170KB/s）：
#   export HF_ENDPOINT=https://hf-mirror.com
#   export HF_HUB_DISABLE_XET=1
# bert-base-uncased 已按本地目录落盘（非 HF blob 布局）：
#   cache/hf/bert-base-uncased/，pin revision 86b5e0934494bd15c9632b12f734a8a67f723594
#   代码中 from_pretrained("cache/hf/bert-base-uncased") 即离线可用

# torchaudio wav2vec2 权重（forced_align 用，走 torch.hub 下载）
export TORCH_HOME="$CUMCM_ROOT/cache/torch"

# Py-Feat 权重默认落 ~/.py-feat（无环境变量时用软链接收口，装好后执行一次）：
#   ln -s "$CUMCM_ROOT/cache/py-feat" "$HOME/.py-feat"

# 首次下载完成、revision 验证通过后打开，防止后续静默联网拉新版本：
# export HF_HUB_OFFLINE=1

# bert-base-uncased model.safetensors sha256 已验证（vs 官方 LFS pointer）：
# 68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
