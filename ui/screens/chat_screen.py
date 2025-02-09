from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.metrics import dp
from ui.widgets.chat_bubble import UserBubble, AIChatBubble, StreamingBubble, LoadingBubble
import logging
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, BooleanProperty
from datetime import datetime
from kivy.app import App
from kivy.core.window import Window
import json
import os
from ui.widgets.history_item import HistoryItem
from kivy.utils import platform
from kivy.uix.spinner import Spinner
from functools import partial
from threading import Thread
from ui.widgets.loading_spinner import LoadingSpinner

logger = logging.getLogger(__name__)

class ChatScreen(RelativeLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_dark_theme = False
        self._sidebar_open = False
        self._setup_event_handlers()
        self.current_conversation_id = None
        self.conversations = {}
        
        # 设置最小窗口大小
        Window.minimum_width = dp(800)
        Window.minimum_height = dp(600)
        
        # 创建加载动画
        self.loading_spinner = LoadingSpinner()
        
        # 延迟加载数据
        Clock.schedule_once(self._init_data, 0.1)
        
        # 延迟加载历史对话
        Clock.schedule_once(lambda dt: self._load_conversations(), 0.5)
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        self.register_event_type('on_message_sent')
        self.register_event_type('on_message_received')
        self.register_event_type('on_send')
    
    def toggle_theme(self):
        self._is_dark_theme = not self._is_dark_theme
        # TODO: 实现主题切换逻辑
    
    def toggle_sidebar(self):
        """优化侧边栏切换动画"""
        print("Toggle sidebar called")  # 添加调试输出
        sidebar = self.ids.sidebar
        print(f"Current sidebar state: {sidebar.is_open}")  # 添加调试输出
        sidebar.toggle()
        self._sidebar_open = sidebar.is_open
        print(f"New sidebar state: {sidebar.is_open}")  # 添加调试输出
    
    def _handle_send_message(self, instance, message):
        if not message or not message.strip():
            return
            
        message = message.strip()
        
        # 添加用户消息
        self.add_message(message, is_user=True)
        self._save_message(message, is_user=True)
        
        # 清空输入框并禁用发送按钮
        self.ids.chat_input.text = ''
        self.ids.send_button.enabled = False
        
        # 保存当前对话ID
        current_id = self.current_conversation_id
        
        # 显示加载动画
        self.loading_spinner.opacity = 0
        self.ids.chat_history.add_widget(self.loading_spinner)
        anim = Animation(opacity=1, duration=0.2)
        anim.start(self.loading_spinner)
        self.loading_spinner.start_spinning()
        self._scroll_to_bottom()
        
        # 在新线程中获取AI响应
        def get_response():
            try:
                response, history = App.get_running_app().generate_response(message)
                
                # 确保仍在同一个对话中
                def add_response(dt):
                    if self.current_conversation_id == current_id:
                        # 移除加载动画
                        if self.loading_spinner in self.ids.chat_history.children:
                            self.loading_spinner.stop_spinning()
                            self.ids.chat_history.remove_widget(self.loading_spinner)
                        
                        # 添加AI响应
                        self.add_message(response, is_user=False)
                        self._save_message(response, is_user=False)
                
                Clock.schedule_once(add_response, 0)
                
            except Exception as e:
                logger.error(f"获取响应失败: {str(e)}")
        
        Thread(target=get_response, daemon=True).start()
    
    def _load_conversations(self):
        """加载历史对话"""
        try:
            # 确保历史对话目录存在
            if not os.path.exists('conversations'):
                os.makedirs('conversations')
                return
            
            # 清空现有历史记录
            self.ids.sidebar.history_list.clear_widgets()
            self.conversations = {}
            
            # 读取所有对话文件并按时间排序
            conversation_files = []
            for file_name in os.listdir('conversations'):
                if file_name.endswith('.json'):
                    file_path = os.path.join('conversations', file_name)
                    conversation_files.append((
                        file_name,
                        os.path.getmtime(file_path)  # 获取文件修改时间
                    ))
            
            # 按时间倒序排序
            conversation_files.sort(key=lambda x: x[1], reverse=True)
            
            # 加载对话
            for file_name, _ in conversation_files:
                conversation_id = file_name[:-5]  # 移除 .json 后缀
                try:
                    with open(f'conversations/{file_name}', 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 获取第一条用户消息作为标题
                        first_message = None
                        for msg in data.get('messages', []):
                            if msg.get('role') == 'user':
                                first_message = msg.get('content')
                                break
                        
                        if not first_message:
                            first_message = "新对话"
                        
                        # 使用创建时间作为时间戳
                        timestamp = data.get('created_at', '')
                        
                        # 添加到历史记录
                        self._add_conversation_to_history(
                            text=first_message,
                            conversation_id=conversation_id,
                            timestamp=timestamp
                        )
                        
                except Exception as e:
                    logger.error(f"加载对话 {conversation_id} 失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"加载历史对话失败: {str(e)}")
    
    def _add_conversation_to_history(self, text, conversation_id=None, timestamp=None):
        """添加对话到历史记录"""
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        history_item = HistoryItem(
            text=text,
            conversation_id=conversation_id,
            timestamp=timestamp
        )
        
        self.ids.sidebar.history_list.add_widget(history_item)
        self.conversations[conversation_id] = {
            'text': text,
            'timestamp': timestamp
        }
    
    def _save_conversations(self):
        """保存对话到缓存"""
        try:
            cache_file = os.path.join(self.data_dir, 'conversations_cache.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")
    
    def _save_message(self, text, is_user=True):
        """保存消息到当前对话"""
        try:
            # 只在第一条用户消息时创建新对话
            if not self.current_conversation_id and is_user:
                self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
                # 只在创建新对话时添加到历史记录
                self._add_conversation_to_history(text, self.current_conversation_id)
            
            # 如果没有当前对话ID，不保存消息
            if not self.current_conversation_id:
                return
            
            # 确保conversations目录存在
            if not os.path.exists('conversations'):
                os.makedirs('conversations')
            
            file_path = f'conversations/{self.current_conversation_id}.json'
            
            # 读取现有对话或创建新对话
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
            else:
                conversation = {
                    'id': self.current_conversation_id,
                    'title': text[:20] + ('...' if len(text) > 20 else ''),
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'messages': []
                }
            
            # 添加新消息
            message = {
                'role': 'user' if is_user else 'assistant',
                'content': text,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            conversation['messages'].append(message)
            
            # 保存对话
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存消息失败: {str(e)}")
    
    def start_new_conversation(self):
        """开始新对话"""
        self.current_conversation_id = None
        self.ids.chat_history.clear_widgets()
        self._update_history_selection()
    
    def load_conversation(self, conversation_id):
        """加载历史对话"""
        try:
            # 如果有加载动画，先移除
            if self.loading_spinner in self.ids.chat_history.children:
                self.loading_spinner.stop_spinning()
                self.ids.chat_history.remove_widget(self.loading_spinner)
            
            file_path = f'conversations/{conversation_id}.json'
            if not os.path.exists(file_path):
                logger.error(f"对话文件不存在: {file_path}")
                return
            
            # 清空当前聊天区域
            self.ids.chat_history.clear_widgets()
            self.current_conversation_id = conversation_id
            
            # 加载对话内容
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            
            # 恢复消息
            for msg in conversation.get('messages', []):
                is_user = msg.get('role') == 'user'
                bubble = UserBubble(text=msg.get('content', '')) if is_user else AIChatBubble(text=msg.get('content', ''))
                bubble.timestamp = msg.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                bubble.opacity = 0
                self.ids.chat_history.add_widget(bubble)
                
                # 创建渐入动画
                anim = Animation(opacity=1, duration=0.2, transition='out_quad')
                anim.start(bubble)
            
            # 更新选中状态
            self._update_history_selection()
            # 滚动到底部
            self._scroll_to_bottom()
            
        except Exception as e:
            logger.error(f"加载对话失败: {str(e)}")
    
    def _update_history_selection(self):
        """更新历史记录的选中状态"""
        sidebar = self.ids.sidebar
        if hasattr(sidebar, 'history_list'):
            for child in sidebar.history_list.children:
                child.selected = (child.conversation_id == self.current_conversation_id)
    
    def on_message_sent(self, message: str) -> None:
        """消息发送事件处理"""
        logger.info(f"消息已发送: {message}")
    
    def on_message_received(self, message: str) -> None:
        """消息接收事件处理"""
        logger.info(f"收到新消息: {message}")
    
    def add_message(self, text, is_user=True):
        """添加消息到聊天区域"""
        try:
            # 创建消息气泡
            bubble = UserBubble(text=text) if is_user else AIChatBubble(text=text)
            bubble.timestamp = datetime.now().strftime("%H:%M")
            bubble.opacity = 0
            
            # 添加到聊天区域
            self.ids.chat_history.add_widget(bubble)
            
            # 创建渐入动画
            anim = Animation(opacity=1, duration=0.2)
            anim.start(bubble)
            
            # 滚动到底部
            self._scroll_to_bottom()
            
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
    
    def _scroll_to_bottom(self, *args):
        """优化滚动到底部的逻辑"""
        def scroll_anim(*_):
            scroll = self.ids.chat_scroll
            if scroll.height < scroll.children[0].height:
                Animation(
                    scroll_y=0, 
                    duration=0.2, 
                    transition='out_quad'
                ).start(scroll)

        # 延迟执行滚动动画，等待布局更新完成
        Clock.schedule_once(scroll_anim, 0.1)

    def on_send(self, text):
        """处理发送消息事件"""
        if not text or not text.strip():
            return
        self._handle_send_message(None, text)

    def _on_window_resize(self, instance, width, height):
        """处理不同设备的窗口大小变化"""
        # 计算缩放比例
        scale_width = width / float(dp(1280))
        scale_height = height / float(dp(720))
        
        # 根据设备类型调整缩放
        if platform in ['android', 'ios']:
            scale = min(scale_width, scale_height) * 1.5
        else:
            scale = min(scale_width, scale_height) * 1.2
        
        scale = max(scale, 0.8)
        
        # 调整侧边栏
        if self._sidebar_open:
            new_width = min(dp(350) * scale, width * 0.3)
            self.ids.sidebar.width = new_width
        
        # 更新控件尺寸
        self._update_sizes(scale)
        
        # 确保滚动到底部
        Clock.schedule_once(self._scroll_to_bottom, 0.1)

    def _update_sizes(self, scale):
        try:
            # 安全地处理子元素
            if hasattr(self.ids, 'chat_history'):
                children = list(self.ids.chat_history.children)
                for child in children:
                    # 对每个子元素进行处理
                    if isinstance(child, (UserBubble, AIChatBubble)):
                        child.size_hint_x = 0.65
                        child.padding = [dp(25) * scale, dp(18) * scale]
        except Exception as e:
            logger.error(f"更新尺寸时出错: {e}")

    def delete_conversation(self, conversation_id):
        """删除对话"""
        try:
            # 删除文件
            file_path = f'conversations/{conversation_id}.json'
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # 如果删除的是当前对话，清空聊天区域并开始新对话
            if conversation_id == self.current_conversation_id:
                self.ids.chat_history.clear_widgets()
                self.current_conversation_id = None
            
            # 从侧边栏移除对话项
            for child in self.ids.sidebar.history_list.children[:]:  # 使用切片创建副本
                if isinstance(child, HistoryItem) and child.conversation_id == conversation_id:
                    self.ids.sidebar.history_list.remove_widget(child)
            
            logger.info(f"成功删除对话: {conversation_id}")
            
        except Exception as e:
            logger.error(f"删除对话失败: {str(e)}")

    def _check_and_fix_conversations_file(self):
        """检查并修复对话文件"""
        file_path = os.path.join(self.data_dir, 'conversations.json')
        try:
            # 如果文件存在但为空或损坏，创建新的空文件
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # 文件为空
                        self._create_empty_conversations_file()
                    else:
                        try:
                            json.loads(content)  # 测试JSON是否有效
                        except json.JSONDecodeError:
                            logger.error("JSON文件损坏，创建新文件")
                            self._create_empty_conversations_file()
            else:
                self._create_empty_conversations_file()
        except Exception as e:
            logger.error(f"检查对话文件时出错: {str(e)}")
            self._create_empty_conversations_file()

    def _create_empty_conversations_file(self):
        """创建空的对话文件"""
        file_path = os.path.join(self.data_dir, 'conversations.json')
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            logger.info("创建了新的空对话文件")
        except Exception as e:
            logger.error(f"创建空对话文件失败: {str(e)}")

    def _init_data(self, dt):
        """延迟初始化数据"""
        # 创建数据目录
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 检查并修复对话文件
        self._check_and_fix_conversations_file()
        
        # 加载历史对话
        self._load_conversations()
        
        # 监听窗口大小变化
        Window.bind(on_resize=self._on_window_resize)