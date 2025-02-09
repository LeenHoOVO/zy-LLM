from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, BooleanProperty, NumericProperty
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.utils import platform
from kivy.animation import Animation
from kivy.clock import Clock
from datetime import datetime

class Sidebar(BoxLayout):
    history_list = ObjectProperty(None)
    is_open = BooleanProperty(False)
    _current_width = NumericProperty(0)
    width = NumericProperty(dp(0))
    width_max = NumericProperty(dp(0))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_x = None
        self.size_hint_y = 1
        self._current_width = dp(0)
        self.opacity = 0
        self._target_width = dp(300)
        
        # 更新最大宽度计算
        def update_width_max(*args):
            scale = min(Window.width / float(dp(1280)), Window.height / float(dp(720))) * 1.3
            target_width = min(dp(300) * scale, Window.width * 0.25)  # 限制最大宽度为25%
            if self.width_max != target_width:
                self.width_max = target_width
                if self.is_open:
                    self._current_width = target_width
        
        Window.bind(size=update_width_max)
        update_width_max()
    
    def toggle(self):
        """优化切换动画"""
        target_width = self.width_max if not self.is_open else dp(0)
        
        # 创建组合动画
        anim = Animation(
            _current_width=target_width,
            duration=0.25,
            t='out_expo'
        )
        
        # 添加透明度动画
        if not self.is_open:
            self.opacity = 0
            anim &= Animation(opacity=1, duration=0.2)
        else:
            anim &= Animation(opacity=0, duration=0.15)
        
        # 启动动画
        anim.start(self)
        self.is_open = not self.is_open 

    def add_conversation(self, text, conversation_id, timestamp=None):
        """添加新的对话到历史记录"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            # 格式化时间戳
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
                
        # 限制文本长度
        if len(text) > 50:
            text = text[:47] + "..."
            
        from ui.widgets.history_item import HistoryItem
        item = HistoryItem(
            text=text,
            timestamp=timestamp,
            conversation_id=conversation_id
        )
        self.history_list.add_widget(item, index=0)  # 新对话添加到顶部 