import base64
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import PhotoImage
from tkinter import filedialog, Menu
from tkinter import ttk, messagebox

from pynput import keyboard

import sound_channel
from b64_encoded_files import icon_base64
from sound_channel import Event, EvtKeys

# 版本号
VERSION = "0.4.0"
channel_base = sound_channel.SoundChannelBase()


def get_event(event_queue, timeout, attempts, expected_key=None):
    """尝试从事件队列中获取事件。
    Args:
        event_queue: 事件队列的基础通道。
        timeout (int): 获取事件的超时时间。
        attempts (int): 获取事件的尝试次数。
        expected_key: 如果提供，则只返回匹配的事件。
    Returns:
        Event: 获取到的事件，如果未获取到则返回None。
    """
    expected_set = set(expected_key)
    for _ in range(attempts):
        try:
            event: Event = event_queue.get(timeout=timeout)
            if expected_key is None or event.key in expected_set:
                return event
        except queue.Empty:
            pass
    return None


class MessageBlock(tk.Frame):
    def __init__(self, master, message, local=True):
        super().__init__(master, bd=1, relief=tk.SOLID, padx=5, pady=5, bg="white")
        self.message = message

        self.text = tk.Text(self, wrap=tk.WORD, height=3, width=50, bg="white", relief=tk.FLAT)
        if local:
            self.text.insert(tk.END, f"Local: {message}")
        else:
            self.text.insert(tk.END, f"Remote: {message}")
        self.text.config(state=tk.DISABLED)
        self.text.pack(fill=tk.BOTH, expand=True)

        self.context_menu = Menu(self, tearoff=0)
        self.context_menu.add_command(label="Copy All", command=self.copy_content)

        self.bind("<Button-3>", self.show_context_menu)
        self.text.bind("<Button-3>", self.show_context_menu)

    def show_context_menu(self, event):
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def copy_content(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.message)


class FileBlock(tk.Frame):
    def __init__(self, master, file_path, file_size=None):
        super().__init__(master, bd=1, relief=tk.SOLID, padx=5, pady=5, bg="white")
        if file_path is None:
            self.desc = "Receive "
            self.file_path = ""
            self.file_name = ""
            self.file_size = ""
        else:
            self.desc = "Send "
            self.file_path = os.path.abspath(file_path)
            self.file_name = os.path.basename(self.file_path)
            if file_size is None:
                self.file_size = os.path.getsize(self.file_path)
            else:
                self.file_size = file_size
        self.start_time = None
        self.is_sender = False
        self.is_done = False
        self.is_cancelled = False
        self.is_failed = False

        self.info_frame = tk.Frame(self, bg="white")
        self.info_frame.pack(fill=tk.X)

        self.file_label = tk.Label(self.info_frame, text=self.desc + self.file_name, bg="white")
        self.file_label.pack(side=tk.LEFT)

        self.size_label = tk.Label(self.info_frame, text="", bg="white")
        self.size_label.pack(side=tk.LEFT, padx=(10, 0))

        self.time_label = tk.Label(self.info_frame, text="", bg="white")
        self.time_label.pack(side=tk.RIGHT)

        self.progress_frame = tk.Frame(self, bg="white")
        self.progress_frame.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cancel_button = tk.Button(self.progress_frame, text="Cancel", command=self.cancel_transfer)
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0))

        self.open_folder_button = tk.Button(self.progress_frame, text="OpenFolder", command=self.open_folder)
        self.open_folder_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.open_folder_button.config(state=tk.DISABLED)

        if self.file_path:
            self.is_sender = True
            event_queue = channel_base.send_event_queue
        else:
            event_queue = channel_base.recv_event_queue
        self.start_transfer(event_queue)

    @property
    def is_finished(self):
        return self.is_done or self.is_failed or self.is_cancelled

    def start_transfer(self, event_queue):
        if self.is_finished:
            return
        # 尝试获取 SEND_FILE_START 事件
        event: Event = get_event(event_queue, timeout=1, attempts=10,
                                 expected_key=[EvtKeys.SEND_FILE_START, EvtKeys.RECV_FILE_START])
        if event:
            if event.key == EvtKeys.RECV_FILE_START:
                self.file_name = event.value
                self.file_label.config(text=self.desc + self.file_name)
                self.file_size = event.o1
            estimate_time = event.o2
            self.start_time = time.time()
            threading.Thread(target=self.simulate_transfer, args=(estimate_time, event_queue), daemon=True).start()

    def set_transfer_freeze(self, desc, color):
        # 冻结当前进度
        current_progress = self.progress["value"]
        # 在描述中添加 "Cancelled"
        current_text = self.file_label.cget("text")
        self.file_label.config(text=f"{current_text} - {desc}")
        # 将进度条变成灰色
        style = ttk.Style()
        style.configure("Failed.Horizontal.TProgressbar", background=color)
        self.progress.configure(style="Failed.Horizontal.TProgressbar")
        # 更新进度标签
        current_size = int(self.file_size * (current_progress / 100))
        size_text = f"{self.format_size(current_size)}/{self.format_size(self.file_size)} - {desc}"
        self.size_label.config(text=size_text)
        # 禁用打开文件夹按钮
        self.open_folder_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)

    def cancel_transfer(self, inform_remote=True):
        if self.is_finished:
            return
        self.is_cancelled = True
        if inform_remote:
            if self.is_sender:
                channel_base.negot_remote_cancel_recv(self.file_name)
                channel_base.cancel_data_sending()
            else:
                channel_base.negot_remote_cancel_send(self.file_name)
                channel_base.cancel_data_receiving()
        self.set_transfer_freeze("Cancelled", "gray")

    def set_transfer_failed(self):
        if self.is_finished:
            return
        self.is_failed = True
        self.set_transfer_freeze("Failed", "pink")

    def simulate_transfer(self, estimate_time, event_queue):
        duration = 0.05
        loops = int(estimate_time / duration)
        for i in range(loops):
            if self.is_finished:
                return
            progress_val = i * 99 / loops
            try:
                if i < loops - 1:
                    event: Event = event_queue.get(timeout=duration)
                else:
                    event: Event = event_queue.get()
                if event.key == EvtKeys.SEND_FINISH:
                    progress_val = 100
                    self.is_done = True
                    self.cancel_button.config(state=tk.DISABLED)
                elif event.key == EvtKeys.RECV_FILE_FINISH:
                    progress_val = 100
                    self.is_done = True
                    self.cancel_button.config(state=tk.DISABLED)
                    self.file_path = event.value
                elif event.key == EvtKeys.FILE_FAIL:
                    self.set_transfer_failed()
                    break
                elif event.key == EvtKeys.FILE_CANCEL:
                    self.cancel_transfer(inform_remote=False)
                    break
            except queue.Empty:
                pass
            self.progress["value"] = progress_val
            current_size = int(self.file_size * (progress_val / 100))
            elapsed_time = round(time.time() - self.start_time, 1)
            self.master.after(0, self.update_info, current_size, elapsed_time)
            if progress_val == 100:
                self.master.after(0, self.transfer_complete)
                break

    def update_info(self, current_size, elapsed_time):
        size_text = f"{self.format_size(current_size)}/{self.format_size(self.file_size)}"
        self.size_label.config(text=size_text)
        self.time_label.config(text=f"{elapsed_time}s")

    def transfer_complete(self):
        self.open_folder_button.config(state=tk.NORMAL)
        # 这里可以添加实际的文件传输逻辑

    def open_folder(self):
        if not os.path.exists(self.file_path):
            messagebox.showerror("Error", f"File not exist: {self.file_name}")
            return

        folder_path = os.path.dirname(self.file_path)

        if os.name == 'nt':  # Windows
            subprocess.Popen(f'explorer /select,"{self.file_path}"')
        elif os.name == 'posix':  # macOS and Linux
            if sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', '-R', self.file_path])
            else:  # Linux
                try:
                    subprocess.Popen(['xdg-open', folder_path])
                except:
                    os.startfile(folder_path)
        else:
            os.startfile(folder_path)

    @staticmethod
    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


class AudioSetDialog(tk.Toplevel):
    def __init__(self, master, kbps):
        self.master = master
        self.kbps = kbps
        self.devices_released = False
        super().__init__()

        self.title("音频设备选择")
        self.geometry("400x400")

        self.selected_input = tk.StringVar()
        self.selected_output = tk.StringVar()

        # 创建一个容器框架来包含Canvas和按钮
        container_frame = ttk.Frame(self)
        container_frame.pack(fill="both", expand=True)

        # 创建Canvas框架来容纳设备列表
        canvas_frame = ttk.Frame(container_frame)
        canvas_frame.pack(fill="both", expand=True)

        # 创建主Canvas和滚动条
        main_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=main_canvas.yview)

        # 创建一个框架来包含所有内容
        content_frame = ttk.Frame(main_canvas)

        # 配置Canvas
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # 创建两个LabelFrame分别用于输入和输出设备
        self.input_frame = ttk.LabelFrame(content_frame, text="输入设备（麦克风）", padding=10)
        self.input_frame.pack(fill="x", padx=10, pady=5)

        self.output_frame = ttk.LabelFrame(content_frame, text="输出设备（扬声器）", padding=10)
        self.output_frame.pack(fill="x", padx=10, pady=5)

        self.refresh_dialog()
        # 放置滚动条和Canvas
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        # 在Canvas中创建窗口
        canvas_frame = main_canvas.create_window((0, 0), window=content_frame, anchor="nw")

        # 添加确认和取消按钮（在容器框架底部）
        button_frame = ttk.Frame(container_frame)
        button_frame.pack(fill="x", padx=10, pady=5, side="bottom")

        ttk.Button(
            button_frame,
            text="确认",
            command=self.on_dialog_confirm
        ).pack(side="right", padx=5)

        ttk.Button(
            button_frame,
            text="取消",
            command=self.destroy
        ).pack(side="right", padx=5)

        ttk.Button(
            button_frame,
            text="刷新",
            command=self.on_dialog_refresh
        ).pack(side="left", padx=5)

        # 配置Canvas滚动区域
        def configure_scroll_region(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))

        def configure_canvas_width(event):
            main_canvas.itemconfig(canvas_frame, width=event.width)

        content_frame.bind("<Configure>", configure_scroll_region)
        main_canvas.bind("<Configure>", configure_canvas_width)

        # 绑定鼠标滚轮
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        # 设置弹窗为模态
        self.transient(self.master)
        self.grab_set()

    def refresh_dialog(self):
        # 清除现有的设备列表
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        for widget in self.output_frame.winfo_children():
            widget.destroy()
        # 添加输入设备选项
        input_devices, output_devices = sound_channel.get_audio_devices()
        # sound_channel.print_audio_devices()
        for device in input_devices:
            radio = ttk.Radiobutton(
                self.input_frame,
                text=f"{device['name']}\n采样率: {device['sample_rate']}Hz, 通道数: {device['channels']}" + (
                    ", 默认设备" if device['default'] else ""),
                variable=self.selected_input,
                value=str(device['index'])
            )
            radio.pack(anchor='w', pady=2)
            if device['index'] == sound_channel.device_indexes[0]:
                self.selected_input.set(str(device['index']))
            elif sound_channel.device_indexes[0] is None and device['default']:
                sound_channel.device_indexes[0] = device['index']
                self.selected_input.set(str(device['index']))

        # 添加输出设备选项
        for device in output_devices:
            radio = ttk.Radiobutton(
                self.output_frame,
                text=f"{device['name']}\n采样率: {device['sample_rate']}Hz, 通道数: {device['channels']}" + (
                    ", 默认设备" if device['default'] else ""),
                variable=self.selected_output,
                value=str(device['index'])
            )
            radio.pack(anchor='w', pady=2)
            if device['index'] == sound_channel.device_indexes[1]:
                self.selected_output.set(str(device['index']))
            elif sound_channel.device_indexes[1] is None and device['default']:
                sound_channel.device_indexes[1] = device['index']
                self.selected_output.set(str(device['index']))

    def on_dialog_confirm(self):
        channel_base.reload_send_speed(self.kbps, int(self.selected_input.get()), int(self.selected_output.get()))
        self.devices_released = False
        self.destroy()

    def on_dialog_refresh(self):
        channel_base.release_all_devices()
        self.devices_released = True
        self.refresh_dialog()

    def destroy(self):
        if self.devices_released:
            channel_base.reload_default_devices()
        super().destroy()


class ChatInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Sound Channel v{VERSION}")
        self.geometry("600x400")
        self.selected_file = None

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top_frame = tk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # 添加两个下拉框
        dropdown1_frame = tk.Frame(top_frame)
        dropdown1_frame.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(dropdown1_frame, text="Send:").pack(side=tk.LEFT)
        self.dropdown1 = ttk.Combobox(dropdown1_frame, values=list(sound_channel.RATES_DESC_TO_IDX.keys()), width=10)
        self.dropdown1.pack(side=tk.LEFT, padx=(0, 5))
        self.dropdown1.set(sound_channel.RATES_IDX_TO_DESC.get(sound_channel.INIT_SEND_kbps))
        self.dropdown1.bind("<<ComboboxSelected>>", self.on_dropdown1_select)

        dropdown2_frame = tk.Frame(top_frame)
        dropdown2_frame.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(dropdown2_frame, text="Recv:").pack(side=tk.LEFT)
        self.dropdown2 = ttk.Combobox(dropdown2_frame, values=list(sound_channel.RATES_DESC_TO_IDX.keys()), width=10)
        self.dropdown2.pack(side=tk.LEFT, padx=(0, 5))
        self.dropdown2.set(sound_channel.RATES_IDX_TO_DESC.get(sound_channel.INIT_RECV_kbps))
        self.dropdown2.bind("<<ComboboxSelected>>", self.on_dropdown2_select)

        self.handshake_button = tk.Button(top_frame, text="Handshake", command=self.handshake)
        self.handshake_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_button = tk.Button(top_frame, text="ClearScreen", command=self.confirm_clear_history)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.file_button = tk.Button(top_frame, text="SelectFile", command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=(0, 5))

        self.setting_button = tk.Button(top_frame, text="AudioSet", command=self.show_device_dialog)
        self.setting_button.pack(side=tk.LEFT)

        main_frame = tk.Frame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.history_canvas = tk.Canvas(main_frame, bg="#f0f0f0")
        self.history_canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.history_canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.history_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.history_frame = tk.Frame(self.history_canvas, bg="#f0f0f0")
        self.history_canvas.create_window((0, 0), window=self.history_frame, anchor="nw")

        self.history_frame.bind("<Configure>", self.on_frame_configure)
        self.history_canvas.bind("<Configure>", self.on_canvas_configure)

        self.input_frame = tk.Frame(main_frame)
        self.input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.input_field = tk.Text(self.input_frame, height=3, wrap=tk.WORD)
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_field.bind("<KeyPress>", self.on_input_key_press)
        self.input_field.bind("<KeyRelease>", self.on_input_key_release)
        self.input_field.bind("<Control-Return>", self.send_with_shortcut)

        self.send_button = tk.Button(self.input_frame, text="Send\n(Ctrl + Enter)", command=self.send_f)
        self.send_button.grid(row=0, column=1, sticky="ns")

        self.notify_monitor_thread = threading.Thread(target=self.notify_monitor, daemon=True)
        self.notify_monitor_thread.start()

        # 在单独的线程中注册热键
        self.hotkey_listener = self.register_hotkey_listener()

    def on_dropdown1_select(self, event):
        desc = self.dropdown1.get()
        kbps = sound_channel.RATES_DESC_TO_IDX[desc]
        channel_base.negot_remote_recv_speed(kbps)
        channel_base.reload_send_speed(kbps)

    def on_dropdown2_select(self, event):
        desc = self.dropdown2.get()
        kbps = sound_channel.RATES_DESC_TO_IDX[desc]
        channel_base.negot_remote_send_speed(kbps)
        channel_base.reload_recv_speed(kbps)

    def notify_monitor(self):
        while True:
            event: Event = channel_base.notify_event_queue.get()
            if event.key == EvtKeys.NOTIFY_MSG:
                self.add_message_block(event.value, local=False)
                self.scroll_to_bottom()
            elif event.key == EvtKeys.NOTIFY_FILE:
                filename = event.value
                file_size = event.o1
                self.add_file_block(filename, file_size=file_size)
                self.scroll_to_bottom()
            elif event.key == EvtKeys.NOTIFY_NEGOT:
                negot_type = event.value
                kbps = event.o1
                if negot_type == sound_channel.TYPE_SEND:
                    self.dropdown1.set(sound_channel.RATES_IDX_TO_DESC.get(int(kbps)))
                else:
                    self.dropdown2.set(sound_channel.RATES_IDX_TO_DESC.get(int(kbps)))

    def register_hotkey_listener(self):
        listener = keyboard.GlobalHotKeys({
            '<alt>+z': self.hotkey_recv_to_clipboard,
            '<alt>+x': self.hotkey_send_from_clipboard
        })
        listener.start()
        return listener

    def hotkey_recv_to_clipboard(self):
        children = [child for child in self.history_frame.winfo_children() if isinstance(child, MessageBlock)]
        if children:
            self.clipboard_clear()
            self.clipboard_append(children[-1].message)

    def hotkey_send_from_clipboard(self):
        message = self.clipboard_get()
        channel_base.send_message(message)
        self.add_message_block(message)
        self.scroll_to_bottom()

    def on_frame_configure(self, event):
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.history_canvas.itemconfig(self.history_canvas.find_all()[0], width=event.width)

    def send_f(self):
        if self.selected_file:
            channel_base.send_file(file_path=self.selected_file)
            # self.add_file_block(self.selected_file)
            self.reset_input_field()
        else:
            message = self.input_field.get("1.0", tk.END).strip()
            if message:
                channel_base.send_message(message)
                self.add_message_block(message)
                self.input_field.delete("1.0", tk.END)
        self.scroll_to_bottom()

    def add_message_block(self, message, local=True):
        block = MessageBlock(self.history_frame, message, local=local)
        block.pack(fill=tk.X, padx=5, pady=2)

    def add_file_block(self, file_path, file_size=None):
        block = FileBlock(self.history_frame, file_path, file_size=file_size)
        block.pack(fill=tk.X, padx=5, pady=2)

    def scroll_to_top(self):
        self.history_canvas.update_idletasks()
        self.history_canvas.yview_moveto(0)

    def scroll_to_bottom(self):
        self.history_canvas.update_idletasks()
        self.history_canvas.yview_moveto(1)

    def send_with_shortcut(self, event):
        self.send_f()
        return 'break'

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_file = file_path
            file_name = os.path.basename(file_path)
            self.input_field.delete("1.0", tk.END)
            self.input_field.insert("1.0", f"Selected file: {file_name}")
            self.input_field.config(bg='lightblue')

    def show_device_dialog(self):
        desc = self.dropdown1.get()
        kbps = sound_channel.RATES_DESC_TO_IDX[desc]
        dialog = AudioSetDialog(self, kbps)
        self.wait_window(dialog)

    def reset_input_field(self):
        self.selected_file = None
        self.input_field.delete("1.0", tk.END)
        self.input_field.config(bg='white')

    def on_input_key_press(self, event):
        if self.selected_file:
            if event.keysym == 'BackSpace':
                self.reset_input_field()
                return 'break'
            elif event.keysym != 'Return':
                return 'break'

    def on_input_key_release(self, event):
        if self.selected_file:
            self.input_field.delete("1.0", tk.END)
            self.input_field.insert("1.0", f"Selected file: {os.path.basename(self.selected_file)}")
            self.input_field.config(bg='lightblue')

    def handshake(self):
        channel_base.negot_start()

    def confirm_clear_history(self):
        if messagebox.askyesno("Confirm", "Are you sure clear history"):
            self.clear_history()

    def clear_history(self):
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        self.scroll_to_top()

    def destroy(self):
        self.hotkey_listener.stop()
        # 在这里添加其他线程的销毁动作
        channel_base.terminate()
        # 调用父类的 destroy 方法
        super().destroy()


# 设置窗口图标
chat_interface = ChatInterface()
# 解码base64数据
icon_data = base64.b64decode(icon_base64)
# 创建PhotoImage对象
icon = PhotoImage(data=icon_data)
chat_interface.tk.call('wm', 'iconphoto', chat_interface._w, icon)
chat_interface.mainloop()
