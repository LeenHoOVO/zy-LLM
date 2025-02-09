from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp

class CustomTextInput(TextInput):
    def __init__(self, **kwargs):
        # 设置默认属性
        kwargs.setdefault('background_color', [0, 0, 0, 0])  # 透明背景
        kwargs.setdefault('cursor_color', [0.3, 0.7, 1, 1])  # 光标颜色
        kwargs.setdefault('foreground_color', [0.2, 0.2, 0.2, 1])  # 文本颜色
        kwargs.setdefault('hint_text_color', [0.6, 0.6, 0.6, 1])  # 提示文字颜色
        kwargs.setdefault('padding', [dp(20), dp(12)])  # 增大内边距
        kwargs.setdefault('font_name', 'SimSun')
        kwargs.setdefault('font_size', dp(16))
        kwargs.setdefault('cursor_width', '2sp')
        
        super().__init__(**kwargs)
        
        # 设置行高实现垂直居中
        self.line_height = 1.2
        
        # 初始化画布
        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # 调整背景色
            self._bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(10)]  # 增大圆角
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
    
    def _update_padding(self, instance, value):
        """更新垂直内边距以保持文本垂直居中"""
        self.padding_y = [self.height / 2.0 - (self.line_height * self.font_size) / 2.0, 0]
    
    def insert_text(self, substring, from_undo=False):
        """处理文本输入"""
        return super().insert_text(substring, from_undo=from_undo)
    
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        """处理键盘事件"""
        return super().keyboard_on_key_down(window, keycode, text, modifiers) 