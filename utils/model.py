import os
import sys
from transformers import AutoTokenizer, AutoModel, PretrainedConfig
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

def load_chatglm_model(model_path="ZhipuAI/ChatGLM-6B", device="auto"):
    """
    加载ChatGLM模型
    
    Args:
        model_path: 模型路径，默认使用在线模型
        device: 设备配置，默认auto自动选择
    
    Returns:
        pipeline对象
    """
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True,
            pre_seq_len=None,
            prefix_projection=False
        ).quantize(4).half().cuda()  # 使用半精度
        
        model = model.eval()
        
        def chat_function(inputs):
            text = inputs['text']
            history = inputs.get('history', [])
            response, new_history = model.chat(tokenizer, text, history)
            return {
                'response': response,
                'history': new_history
            }
            
        return chat_function
        
    except Exception as e:
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        raise Exception(f"模型加载失败: {str(e)}")

def generate_response_from_model(pipe, text, history=None):
    """
    生成模型回答
    
    Args:
        pipe: 模型函数
        text: 输入文本
        history: 对话历史，默认为None
    
    Returns:
        tuple: (回答文本, 更新后的历史记录)
    """
    if history is None:
        history = []
        
    try:
        inputs = {
            'text': text,
            'history': history
        }
        result = pipe(inputs)
        
        response = result['response']
        new_history = result['history']
        
        return response, new_history
    except Exception as e:
        raise Exception(f"生成回答失败: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    pipe = load_chatglm_model()
    response, history = generate_response_from_model(pipe, "你好")
    print(f"测试回答：{response}")
