from kivy.uix.widget import Widget
from kivy.properties import NumericProperty
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.graphics import Color, Line
from kivy.lang import Builder

Builder.load_string("""
<LoadingSpinner>:
    size_hint: None, None
    size: dp(30), dp(30)
    pos_hint: {'center_x': 0.5, 'center_y': 0.5}
""")

class LoadingSpinner(Widget):
    alpha = NumericProperty(1.0)  # 用于控制透明度
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = None, None
        self.size = dp(30), dp(30)  # 减小整体大小
        self._lines = []
        self._colors = []
        self._animations = []
        
        with self.canvas:
            # 绘制爻字的三条横线
            for i in range(3):
                color = Color(0.3, 0.7, 1, 1)
                self._colors.append(color)
                
                # 每个爻由两条短线组成
                line_left = Line(width=dp(1.5))  # 减小线条宽度
                line_right = Line(width=dp(1.5))
                self._lines.extend([line_left, line_right])
        
        self.bind(pos=self._update_lines, size=self._update_lines)
        self._update_lines()
    
    def _update_lines(self, *args):
        """更新线条位置"""
        spacing = self.height / 4  # 线条间距
        line_width = self.width * 0.35  # 减小线条长度
        gap = self.width * 0.3  # 中间的间隔
        
        for i in range(3):
            y = self.y + spacing * (i + 1)
            # 左边的线
            self._lines[i*2].points = [
                self.x, y,
                self.x + line_width, y
            ]
            # 右边的线
            self._lines[i*2+1].points = [
                self.right - line_width, y,
                self.right, y
            ]
    
    def start_spinning(self):
        """开始加载动画"""
        # 为每个爻创建交错的动画
        for i, color in enumerate(self._colors):
            anim = (
                Animation(a=0.2, duration=0.6) +  # 降低最小透明度，增加动画时长
                Animation(a=1, duration=0.6)
            )
            anim.repeat = True
            
            # 错开动画开始时间
            Clock.schedule_once(
                lambda dt, a=anim, c=color: a.start(c), 
                i * 0.4  # 增加错开时间
            )
            self._animations.append(anim)
    
    def stop_spinning(self):
        """停止加载动画"""
        for anim in self._animations:
            anim.cancel_all(self)
        self._animations.clear()
        # 重置所有线条的透明度
        for color in self._colors:
            color.a = 1