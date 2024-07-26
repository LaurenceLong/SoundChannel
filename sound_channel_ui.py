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

import keyboard

from b64_encoded_files import icon_base64
from sound_channel import SoundChannelBase, Event, Evt

# 版本号
VERSION = "0.1.0"
channel_base = SoundChannelBase()


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
            self.text.insert(tk.END, f"You: {message}")
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
    def __init__(self, master, file_path):
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
            self.file_size = os.path.getsize(self.file_path)
        self.start_time = None

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

        self.open_folder_button = tk.Button(self.progress_frame, text="OpenFolder", command=self.open_folder)
        self.open_folder_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.open_folder_button.config(state=tk.DISABLED)

        if self.file_path:
            event_queue = channel_base.send_event_queue
        else:
            event_queue = channel_base.recv_event_queue
        self.start_transfer(event_queue)

    def start_transfer(self, event_queue):
        # 尝试获取 SEND_FILE_START 事件
        event: Event = get_event(event_queue, timeout=2, attempts=5,
                                 expected_key=[Evt.SEND_FILE_START, Evt.RECV_FILE_START])
        if event:
            if event.key == Evt.RECV_FILE_START:
                self.file_name = event.value
                self.file_label.config(text=self.desc + self.file_name)
                self.file_size = event.o1
            # 如果获取到了 SEND_FILE_START 事件，尝试获取 SEND_TIME 事件
            event = event_queue.get()  # 这里假设能立即获取到，不需要超时
            if event:
                estimate_time = event.value
                self.start_time = time.time()
                threading.Thread(target=self.simulate_transfer, args=(estimate_time, event_queue), daemon=True).start()

    def simulate_transfer(self, estimate_time, event_queue):
        duration = 0.05
        loops = int(estimate_time / duration)
        for i in range(loops):
            progress_val = i * 99 / loops
            try:
                if i < loops - 1:
                    event: Event = event_queue.get(timeout=duration)
                else:
                    event: Event = event_queue.get()
                if event.key == Evt.SEND_FINISH:
                    progress_val = 100
                elif event.key == Evt.RECV_FILE_FINISH:
                    progress_val = 100
                    self.file_path = event.value
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

        self.handshake_button = tk.Button(top_frame, text="Handshake", command=self.handshake)
        self.handshake_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_button = tk.Button(top_frame, text="ClearScreen", command=self.confirm_clear_history)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.file_button = tk.Button(top_frame, text="SelectFile", command=self.select_file)
        self.file_button.pack(side=tk.LEFT)

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

        input_frame = tk.Frame(main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_field = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_field.bind("<KeyPress>", self.on_input_key_press)
        self.input_field.bind("<KeyRelease>", self.on_input_key_release)
        self.input_field.bind("<Control-Return>", self.send_with_shortcut)

        self.send_button = tk.Button(input_frame, text="Send\n(Ctrl + Enter)", command=self.send_f)
        self.send_button.grid(row=0, column=1, sticky="ns")

        self.notify_monitor_thread = threading.Thread(target=self.notify_monitor, daemon=True)
        self.notify_monitor_thread.start()

        # 在单独的线程中注册热键
        self.hotkey_thread = threading.Thread(target=self.register_hotkey, daemon=True)
        self.hotkey_thread.start()

    def notify_monitor(self):
        while True:
            event: Event = channel_base.notify_event_queue.get()
            if event.key == Evt.NOTIFY_MSG:
                self.add_message_block(event.value, local=False)
            elif event.key == Evt.NOTIFY_FILE:
                self.add_file_block(None)
            self.scroll_to_bottom()

    def register_hotkey(self):
        keyboard.add_hotkey('ctrl+alt+c', self.hotkey_direct_copy)
        keyboard.add_hotkey('ctrl+alt+v', self.hotkey_direct_paste)

    def hotkey_direct_copy(self):
        children = [child for child in self.history_frame.winfo_children() if isinstance(child, MessageBlock)]
        if children:
            self.clipboard_clear()
            self.clipboard_append(children[-1].message)

    def hotkey_direct_paste(self):
        message = self.clipboard_get()
        channel_base.send_message(message)
        self.add_message_block(message)

    def on_frame_configure(self, event):
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.history_canvas.itemconfig(self.history_canvas.find_all()[0], width=event.width)

    def send_f(self):
        if self.selected_file:
            channel_base.send_file(file_path=self.selected_file)
            self.add_file_block(self.selected_file)
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

    def add_file_block(self, file_path):
        block = FileBlock(self.history_frame, file_path)
        block.pack(fill=tk.X, padx=5, pady=2)

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
        self.add_message_block("Handshake initiated.")
        self.scroll_to_bottom()

    def confirm_clear_history(self):
        if messagebox.askyesno("Confirm", "Are you sure clear history"):
            self.clear_history()

    def clear_history(self):
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        self.scroll_to_bottom()

    def destroy(self):
        # 在这里添加其他进程的销毁动作
        channel_base.stop()
        time.sleep(1)
        for process in channel_base.processes:
            try:
                process.kill()  # 或 process.kill()，取决于您希望如何结束进程
            except:
                print(f"Unable to terminate: {process}")
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
