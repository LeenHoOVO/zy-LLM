import os
from gptpdf import parse_pdf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认提示词配置
DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。
3. 内容不要包含在```markdown ```中、段落公式使用适当的形式、行内公式使用适当的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""

DEFAULT_RECT_PROMPT = """图片中用红色框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""

DEFAULT_ROLE_PROMPT = """你是一个专业的易经文档解析器，使用markdown和latex语法输出图片的内容。特别注意保持易经中的卦象符号和专业术语的准确性。
"""

class PDFProcessor:
    def __init__(self, api_key: str = None, base_url: str = "https://www.gptapi.us/v1"):
        """
        初始化 PDF 处理器
        Args:
            api_key: OpenAI API 密钥
            base_url: OpenAI API 基础地址
        """
        self.api_key = api_key
        self.base_url = base_url
        self.prompt_dict = {
            'prompt': DEFAULT_PROMPT,
            'rect_prompt': DEFAULT_RECT_PROMPT,
            'role_prompt': DEFAULT_ROLE_PROMPT
        }
        
    def process_pdf(self, pdf_path: str) -> str:
        """
        处理 PDF 文件并保存为 Markdown
        Args:
            pdf_path: PDF 文件路径
        Returns:
            提取的文本内容
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF文件不存在: {pdf_path}")
                return ""
                
            # 创建输出目录
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用 gptpdf 解析 PDF
            content, images = parse_pdf(
                pdf_path=pdf_path,
                output_dir=str(output_dir),
                prompt=self.prompt_dict,
                api_key=self.api_key,
                base_url=self.base_url,
                model="gpt-4-vision-preview",
                verbose=True,
                gpt_worker=2,
                timeout=120
            )
            
            # 保存为 Markdown 文件
            pdf_name = Path(pdf_path).stem
            markdown_dir = output_dir / "markdown"
            markdown_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = markdown_dir / f"{pdf_name}.md"
            
            # 构建 Markdown 内容
            markdown_content = f"# {pdf_name}\n\n"
            markdown_content += content + "\n\n"
            
            # 如果有图片，添加图片引用
            if images:
                markdown_content += "## 图片\n\n"
                for i, image_path in enumerate(images):
                    rel_path = os.path.relpath(image_path, markdown_dir)
                    markdown_content += f"![图片{i+1}]({rel_path})\n\n"
            
            # 保存 Markdown 文件
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            logger.info(f"Markdown 文件已保存到: {markdown_path}")
            
            return content
            
        except Exception as e:
            logger.error(f"处理PDF文件失败 {pdf_path}: {str(e)}")
            return ""
            
    def get_page_count(self, pdf_path: str) -> int:
        """获取 PDF 页数"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"获取PDF页数失败 {pdf_path}: {str(e)}")
            return 0 