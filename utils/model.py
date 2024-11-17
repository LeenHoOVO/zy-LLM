from transformers import AutoModel, AutoTokenizer
import torch
import os

def load_chatglm_model(model_path="./models/ZhipuAI/ChatGLM-6B"):
    """
    加载chatglm-6b模型
    Args:
        model_path: 模型路径或模型名称
    """
    try:
        print(f"正在加载模型，路径: {model_path}")
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16
        ).eval()
        
        print("模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise

def generate_response_from_model(model, tokenizer, prompt, max_length=2048):
    """
    使用模型生成回答
    Args:
        model: ChatGLM模型
        tokenizer: ChatGLM分词器
        prompt: 输入的提示文本
        max_length: 生成文本的最大长度
    """
    try:
        response, history = model.chat(tokenizer, prompt, history=[])
        return response
    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return f"生成回答失败: {str(e)}"

if __name__ == "__main__":
    # 测试代码
    try:
        model, tokenizer = load_chatglm_model()
        response = generate_response_from_model(model, tokenizer, "你好")
        print(f"测试回答：{response}")
    except Exception as e:
        print(f"测试失败: {str(e)}")
