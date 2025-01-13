from huggingface_hub import snapshot_download
import os

def download_model():
    print("开始下载ChatGLM-6B模型...")
    try:
        # 设置下载路径
        model_dir = os.path.abspath("models/ZhipuAI/ChatGLM-6B")
        os.makedirs(model_dir, exist_ok=True)
        
        # 从Hugging Face下载模型
        model_id = 'THUDM/chatglm-6b'
        cache_dir = model_dir
        
        print(f"正在下载模型到: {model_dir}")
        print("注意：模型大小约为14GB，下载可能需要较长时间...")
        
        downloaded_model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=model_dir,
            local_dir_use_symlinks=False  # 不使用符号链接，直接复制文件
        )
        
        print(f"模型下载完成！保存在: {downloaded_model_path}")
        
    except Exception as e:
        print(f"下载过程中出错: {str(e)}")
        print("\n您可以尝试手动下载：")
        print(f"1. 访问 https://huggingface.co/{model_id}")
        print(f"2. 下载所有文件")
        print(f"3. 将文件放到 {model_dir} 目录下")

if __name__ == "__main__":
    download_model() 