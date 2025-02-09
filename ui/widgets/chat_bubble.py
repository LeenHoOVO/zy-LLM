from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import (
    StringProperty, 
    BooleanProperty, 
    NumericProperty,
    ListProperty,
    ObjectProperty
)
from kivy.metrics import dp
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle, Ellipse
import logging
from kivy.core.window import Window
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageContainer(BoxLayout):
    """消息容器，包含时间戳和气泡"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.spacing = dp(4)  # 时间戳和气泡之间的间距
        self.bind(minimum_height=self.setter('height'))

class ChatBubble(BoxLayout):
    text = StringProperty('')
    bubble_color = ListProperty([1, 1, 1, 1])
    text_color = ListProperty([0, 0, 0, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        
        # 气泡设置
        self.size_hint_x = 0.9  # 增大气泡宽度
        self.padding = [dp(15), dp(10)]  # 调整内边距
        self.bind(
            minimum_height=self.setter('height'),
            pos=self.update_canvas,
            size=self.update_canvas
        )

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(rgba=self.bubble_color)
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(15)]
            )

    def format_text(self, text):
        """格式化文本，处理标题、列表和加粗文本"""
        formatted = []
        lines = text.split('\n')
        
        def process_bold(text):
            """处理文本中的加粗标记"""
            if '**' in text:
                parts = text.split('**')
                formatted_parts = []
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # 奇数部分是需要加粗的
                        formatted_parts.append(f'[b]{part}[/b]')
                    else:
                        formatted_parts.append(part)
                return ''.join(formatted_parts)
            return text
        
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            
            # 处理带编号和 ### 的标题
            if line.startswith('###'):
                title = line.lstrip('#').strip()
                # 处理标题中的加粗
                title = process_bold(title)
                formatted.append(f'[size=22][b]{title}[/b][/size]')
                continue
            
            # 处理带 ** 的编号标题
            if any(line.startswith(f"{i}. ") for i in range(1, 6)):
                number, content = line.split('.', 1)
                content = content.strip()
                
                # 处理标题中的加粗
                if content.strip().startswith('**') and content.strip().endswith('**'):
                    # 如果整个内容都是加粗的，移除 ** 并加粗整行
                    content = content.strip().strip('*')
                    formatted.append(f'[size=22][b]{number}. {content}[/b][/size]')
                else:
                    # 处理内容中的加粗部分
                    content = process_bold(content)
                    formatted.append(f'{number}. {content}')
                continue
            
            # 处理 • 开头的列表项
            if line.strip().startswith(('•', '-')):
                content = line.strip()[1:].strip()
                # 处理列表项中的加粗
                content = process_bold(content)
                formatted.append(f'  • {content}')
                continue
            
            # 处理普通段落中的加粗
            line = process_bold(line)
            formatted.append(line)
        
        return '\n'.join(formatted)

    def on_text(self, instance, value):
        """当文本更新时进行格式化"""
        self.formatted_text = self.format_text(value)

class UserBubble(ChatBubble):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bubble_color = [0.95, 0.95, 0.95, 1]  # 浅灰色背景
        self.text_color = [0.2, 0.2, 0.2, 1]
        self.pos_hint = {'right': 0.98}

class AIChatBubble(ChatBubble):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bubble_color = [0.0, 0.6, 1, 1]  # 蓝色
        self.text_color = [1, 1, 1, 1]
        self.pos_hint = {'x': 0.02}

class StreamingBubble(AIChatBubble):
    """带打字机效果的AI回复气泡"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.full_text = ""
        self.current_text = ""
        self.char_delay = 0.03  # 每个字符的延迟时间
        self.is_streaming = False
        self.stream_event = None
    
    def start_streaming(self, text):
        """开始打字机效果"""
        self.full_text = text
        self.current_text = ""
        self.text = ""
        self.is_streaming = True
        self.stream_next_char()
    
    def stream_next_char(self, *args):
        """显示下一个字符"""
        if not self.is_streaming:
            return
        
        if len(self.current_text) < len(self.full_text):
            self.current_text += self.full_text[len(self.current_text)]
            self.text = self.current_text
            self.stream_event = Clock.schedule_once(self.stream_next_char, self.char_delay)
        else:
            self.is_streaming = False
            self.stream_event = None
    
    def stop_streaming(self):
        """停止打字机效果"""
        if self.stream_event:
            self.stream_event.cancel()
        self.is_streaming = False
        self.text = self.full_text

class LoadingBubble(BoxLayout):
    """加载状态气泡"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (dp(80), dp(40))
        self.pos_hint = {'x': 0.02}
        self._setup_animation()  # 确保调用动画设置
    
    def _setup_animation(self):
        with self.canvas:
            self.dot_colors = [
                Color(0.6, 0.6, 0.6, 1),
                Color(0.6, 0.6, 0.6, 1),
                Color(0.6, 0.6, 0.6, 1)
            ]
            self.dots = [
                Ellipse(
                    pos=(self.center_x + offset, self.center_y - dp(4)),
                    size=(dp(8), dp(8))
                ) for offset in [-dp(20), -dp(4), dp(12)]
            ]

        self.animation_phase = 0
        self.animation_event = Clock.schedule_interval(
            self._animate_dots, 
            0.4
        )
    
    def on_pos(self, *args):
        """更新点的位置"""
        if hasattr(self, 'dots'):
            offsets = [-dp(20), -dp(4), dp(12)]
            for dot, offset in zip(self.dots, offsets):
                dot.pos = (self.center_x + offset, self.center_y - dp(4))

    def _animate_dots(self, dt):
        phases = [
            [0.3, 0.5, 0.3],
            [0.5, 0.3, 0.5],
            [0.3, 0.3, 0.5]
        ]
        for i, color in enumerate(self.dot_colors):
            color.a = phases[self.animation_phase][i]
            
        self.animation_phase = (self.animation_phase + 1) % 3

    def on_parent(self, instance, parent):
        if parent is None and self.animation_event:
            self.animation_event.cancel()