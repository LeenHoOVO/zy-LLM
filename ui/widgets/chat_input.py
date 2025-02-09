from kivy.uix.textinput import TextInput
from kivy.metrics import dp
from kivy.utils import platform
from kivy.graphics import Color, RoundedRectangle

class ChatInput(TextInput):
    def __init__(self, **kwargs):
        # 设置默认属性
        kwargs.setdefault('background_color', [0, 0, 0, 0])  # 透明背景
        kwargs.setdefault('cursor_color', [0.3, 0.7, 1, 1])  # 光标颜色
        kwargs.setdefault('foreground_color', [0.2, 0.2, 0.2, 1])  # 文本颜色
        kwargs.setdefault('hint_text_color', [0.5, 0.5, 0.5, 1])
        kwargs.setdefault('padding', 
            [dp(15), dp(10)] if platform in ('android', 'ios') else [dp(20), dp(15)])
        kwargs.setdefault('font_name', 'SimSun')
        kwargs.setdefault('font_size', dp(14) if platform in ('android', 'ios') else dp(16))
        kwargs.setdefault('multiline', False)
        kwargs.setdefault('cursor_width', '2sp')
        
        super().__init__(**kwargs)
        
        # 设置行高实现垂直居中
        self.line_height = 1.2
        
        # 初始化画布
        with self.canvas.before:
            Color(0.97, 0.97, 0.97, 1)
            self._bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(8) if platform in ('android', 'ios') else dp(10)]
            )
        
        # 绑定位置和大小更新
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )
    
    def _update_rect(self, instance, value):
        """更新背景矩形的位置和大小"""
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
    
    def insert_text(self, substring, from_undo=False):
        """处理文本输入"""
        return super().insert_text(substring, from_undo=from_undo)
    
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        """处理键盘事件"""
        # 处理回车发送
        if keycode[1] == 'enter' and not modifiers:
            # 触发文本验证事件
            self.dispatch('on_text_validate')
            return True
        return super().keyboard_on_key_down(window, keycode, text, modifiers) 