#!/usr/bin/env python3
"""从 OSS 下载模型"""

import oss2
import os
import sys

# OSS 配置 - 必须设置环境变量
access_key_id = os.environ.get("OSS_ACCESS_KEY_ID")
access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")

if not access_key_id or not access_key_secret:
    print("错误: 请设置环境变量 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET")
    sys.exit(1)
endpoint = os.environ.get("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
bucket_name = os.environ.get("OSS_BUCKET", "funasr2")

auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

def download_prefix(prefix, local_dir):
    """下载指定前缀下的所有文件"""
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"下载: {prefix}")
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        if obj.key.endswith("/"):
            continue
        
        relative = obj.key[len(prefix):]
        local_path = os.path.join(local_dir, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        size_mb = obj.size / 1024 / 1024
        print(f"  {relative} ({size_mb:.1f} MB)")
        bucket.get_object_to_file(obj.key, local_path)

if __name__ == "__main__":
    # 下载 FunASR nano 模型
    download_prefix("models/funasr-nano-int8/", "/app/models/funasr-nano-int8")
    
    # 下载 VAD 模型
    download_prefix("models/vad/", "/app/models/vad")
    
    print("\n完成！")