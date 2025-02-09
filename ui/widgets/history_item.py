from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.app import App
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HistoryItem(ButtonBehavior, BoxLayout):
    text = StringProperty('')
    timestamp = StringProperty('')
    conversation_id = StringProperty('')
    selected = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.timestamp:
            # 显示完整日期和时间
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def on_press(self):
        """处理点击事件"""
        app = App.get_running_app()
        if app and app.root:
            app.root.load_conversation(self.conversation_id)
    
    def on_touch_down(self, touch):
        """处理触摸事件"""
        # 检查是否点击了删除按钮
        delete_btn = self.ids.get('delete_btn')
        if delete_btn and delete_btn.collide_point(*touch.pos):
            self.delete_item()
            return True
        return super().on_touch_down(touch)
    
    def delete_item(self, *args):
        """删除历史记录项"""
        app = App.get_running_app()
        if app:
            app.show_delete_confirmation(
                lambda: self._confirm_delete()
            )
    
    def _confirm_delete(self):
        """确认删除后的操作"""
        app = App.get_running_app()
        if app and app.root:
            app.root.delete_conversation(self.conversation_id)
        if self.parent:
            self.parent.remove_widget(self) 