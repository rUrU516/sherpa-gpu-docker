# sherpa-gpu-docker

Docker 镜像，包含 sherpa-onnx GPU 支持和 FunASR nano 模型。

## 构建

GitHub Actions 自动构建（push 到 main 分支触发）。

## 需要的 GitHub Secrets

在仓库设置中添加（Settings → Secrets and variables → Actions）：

- `ALIYUN_REGISTRY_USERNAME` - 阿里云容器镜像服务用户名
- `ALIYUN_REGISTRY_PASSWORD` - 阿里云容器镜像服务密码

## 镜像地址

```
registry.cn-hangzhou.aliyuncs.com/fituer/sherpa-gpu:latest
```

## 在 FC GPU 实例上运行

```bash
# 拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/fituer/sherpa-gpu:latest

# 运行
docker run --gpus all -it \
  -v /path/to/wavs:/app/wavs \
  registry.cn-hangzhou.aliyuncs.com/fituer/sherpa-gpu:latest

# 容器内下载模型
python /app/scripts/download_models.py

# 运行测试
TEST_WAV=/app/wavs/test.wav python /app/scripts/test_funasr.py
```

## 检查 GPU 是否工作

运行时看输出，如果**没有** `Fallback to cpu` 就是 GPU 在工作。