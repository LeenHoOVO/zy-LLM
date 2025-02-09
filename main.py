import logging
from kivy.app import App
from kivy.lang import Builder
from ui.screens.chat_screen import ChatScreen
import os
from kivy.core.text import LabelBase
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.utils import platform
from utils.model import AIModel
from utils.RAG import basic_rag
from utils.graph_rag import GraphRAG
import json
from threading import Thread
from kivy.clock import Clock

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 注册字体r
LabelBase.register(name='SimSun',
                  fn_regular=r'ui\assets\fonts\SIMSUN.TTC')

class ChatApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_history = []
        
        # 确保对话目录存在
        if not os.path.exists('conversations'):
            os.makedirs('conversations')
            
        # 加载历史对话
        self._load_conversations()
        
        try:
            # 初始化AI模型
            self.model = AIModel(
                api_key="sk-180dcaf1fc30403fb009abec50d59b43",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                temperature=1.3,
                max_tokens=2000
            )
            logger.info("AI模型初始化成功")
        except Exception as e:
            logger.error(f"AI模型初始化失败: {str(e)}")
            self.model = None
        
        # 异步加载 GraphRAG
        self.graph_rag = None
        Thread(target=self._load_graph_rag, daemon=True).start()
        
        # 加载 KV 文件
        try:
            kv_file = 'ui/styles/main.kv'
            if not os.path.exists(kv_file):
                logger.error(f"KV文件不存在: {kv_file}")
                raise FileNotFoundError(f"找不到KV文件: {kv_file}")
            Builder.load_file(kv_file)
            logger.info("KV 文件加载成功")
        except Exception as e:
            logger.error(f"KV文件加载失败: {str(e)}")
        
    def build(self):
        # 根据平台设置不同的窗口属性
        if platform != 'android' and platform != 'ios':
            # 桌面端最小尺寸 - 移到 Window.create_window 之后
            Clock.schedule_once(lambda dt: setattr(Window, 'minimum_width', dp(800)), 0)
            Clock.schedule_once(lambda dt: setattr(Window, 'minimum_height', dp(600)), 0)
        else:
            # 移动端设置
            Window.softinput_mode = 'below_target'  # 确保软键盘不会遮挡输入框
        
        return ChatScreen()
        
    def generate_response(self, text):
        """生成AI回复"""
        try:
            # 确保 chat_history 的格式正确
            formatted_history = []
            for msg in self.chat_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    formatted_history.append(msg)
                else:
                    logger.warning(f"跳过格式不正确的历史消息: {msg}")
            
            # 使用格式化后的历史记录
            result = self.model.generate_with_rag(
                query=text,
                history=formatted_history
            )
            
            if isinstance(result, dict) and 'response' in result and 'history' in result:
                self.chat_history = result['history']
                return result['response'], result['history']
            else:
                logger.error("生成的响应格式不正确")
                return "生成响应格式错误", self.chat_history
        except Exception as e:
            logger.error(f"生成响应失败: {str(e)}")
            return f"生成响应时出错: {str(e)}", self.chat_history

    def show_delete_confirmation(self, callback):
        """显示删除确认对话框"""
        content = BoxLayout(orientation='vertical', spacing=dp(20), padding=dp(25))
        
        # 添加确认消息
        content.add_widget(Label(
            text='确定要删除这条记录吗？',
            size_hint_y=0.6,
            font_name='SimSun',
            color=(0.3, 0.3, 0.3, 1),
            font_size=dp(16)
        ))
        
        # 添加按钮布局
        buttons = BoxLayout(size_hint_y=0.4, spacing=dp(20))
        
        # 取消按钮
        cancel_btn = Button(
            text='取消',
            font_name='SimSun',
            size_hint_x=0.5,
            background_normal='',
            background_color=(0.95, 0.95, 0.95, 1),
            color=(0.4, 0.4, 0.4, 1)
        )
        
        # 确认按钮
        confirm_btn = Button(
            text='确定',
            font_name='SimSun',
            size_hint_x=0.5,
            background_normal='',
            background_color=(0.9, 0.3, 0.3, 0.9),
            color=(1, 1, 1, 1)
        )
        
        # 为按钮添加圆角
        for btn in (cancel_btn, confirm_btn):
            with btn.canvas.before:
                Color(rgba=btn.background_color)
                self.rect = RoundedRectangle(
                    pos=btn.pos,
                    size=btn.size,
                    radius=[dp(6)]
                )
            btn.bind(pos=self._update_rect, size=self._update_rect)
        
        buttons.add_widget(cancel_btn)
        buttons.add_widget(confirm_btn)
        content.add_widget(buttons)
        
        # 创建弹窗
        popup = Popup(
            title='确认删除',
            content=content,
            size_hint=(None, None),
            size=(dp(340), dp(200)),
            background='',
            background_color=(0.98, 0.98, 0.98, 1),  # 改回不透明背景
            title_color=(0.3, 0.3, 0.3, 1),
            title_size=dp(18),
            title_align='center',
            separator_height=0,
            auto_dismiss=False
        )
        
        # 绑定按钮事件
        cancel_btn.bind(on_release=popup.dismiss)
        confirm_btn.bind(on_release=lambda x: self.handle_delete_confirmation(popup, callback))
        
        # 添加弹窗背景和阴影
        with popup.canvas.before:
            # 阴影
            Color(0, 0, 0, 0.1)
            RoundedRectangle(
                pos=(popup.x + dp(2), popup.y - dp(2)),
                size=popup.size,
                radius=[dp(12)]
            )
            # 主背景
            Color(0.98, 0.98, 0.98, 1)
            RoundedRectangle(
                pos=popup.pos,
                size=popup.size,
                radius=[dp(12)]
            )
        
        popup.open()
    
    def handle_delete_confirmation(self, popup, callback):
        """处理删除确认"""
        popup.dismiss()
        if callback:
            callback()

    def _update_rect(self, instance, value):
        """更新按钮背景"""
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(rgba=instance.background_color)
            RoundedRectangle(
                pos=instance.pos,
                size=instance.size,
                radius=[dp(6)]
            )

    def _load_graph_rag(self):
        """异步加载 GraphRAG"""
        try:
            self.graph_rag = GraphRAG.load("data/graph_rag")
            logger.info("GraphRAG 加载成功")
        except Exception as e:
            logger.warning(f"GraphRAG 加载失败，将使用基础 RAG: {str(e)}")

    def _load_conversations(self):
        """加载所有历史对话"""
        try:
            conversation_files = sorted(
                [f for f in os.listdir('conversations') if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join('conversations', x)),
                reverse=True
            )
            
            for file_name in conversation_files:
                try:
                    with open(f'conversations/{file_name}', 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        if conversation.get('messages'):
                            first_msg = conversation['messages'][0]['content']
                            timestamp = conversation['created_at']
                            conversation_id = file_name[:-5]  # 移除 .json
                            
                            # 添加到侧边栏
                            Clock.schedule_once(
                                lambda dt, t=first_msg, cid=conversation_id, ts=timestamp: 
                                self.root.ids.sidebar.add_conversation(t, cid, ts)
                            )
                except Exception as e:
                    logger.error(f"加载对话文件失败 {file_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"加载历史对话失败: {str(e)}")

def main():
    if platform == 'win':
        # Windows下修复高DPI显示问题
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        
    app = ChatApp()
    app.run()

if __name__ == "__main__":
    main() 