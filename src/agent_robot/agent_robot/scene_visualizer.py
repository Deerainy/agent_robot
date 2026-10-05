import json
import tkinter as tk
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SceneVisualizer(Node):

    def __init__(self):
        super().__init__('scene_visualizer')

        # 订阅执行器发布的状态
        self.status_subscription = self.create_subscription(
            String,
            '/task_status',
            self.execution_status_callback,
            10
        )
        self.command_publisher = self.create_publisher(
            String,
            '/user_command',
            10
        )

        # 创建可视化窗口
        self.root = tk.Tk()
        self.root.title('EmbodiedPlan - Scene Visualizer')
        self.root.geometry('900x650')
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=900,
            height=520,
            bg='#f3eee5',
            highlightthickness=0
        )
        self.canvas.pack()

        # 用户任务输入区域
        self.command_frame = tk.Frame(
            self.root,
            bg='#eceff1',
            padx=12,
            pady=10
        )
        self.command_frame.pack(fill='x')

        self.command_entry = tk.Entry(
            self.command_frame,
            font=('Arial', 13)
        )
        self.command_entry.pack(
            side='left',
            fill='x',
            expand=True,
            padx=(0, 10),
            ipady=6
        )

        self.command_entry.insert(
            0,
            'Pick up the red apple and place it into the basket.'
        )

        self.send_button = tk.Button(
            self.command_frame,
            text='Send Task',
            font=('Arial', 12, 'bold'),
            bg='#1976d2',
            fg='white',
            activebackground='#1565c0',
            activeforeground='white',
            command=self.send_command,
            padx=18,
            pady=5
        )
        self.send_button.pack(side='right')

        # 按 Enter 也可以发送任务
        self.command_entry.bind(
            '<Return>',
            lambda event: self.send_command()
        )

        self.status_label = tk.Label(
            self.root,
            text='System ready - waiting for task...',
            font=('Arial', 14),
            bg='#263238',
            fg='white',
            anchor='w',
            padx=15
        )
        self.status_label.pack(fill='both', expand=True)

        # 场景状态
        self.apple_state = 'table'
        self.apple_x = 250
        self.apple_y = 350

        self.draw_scene()

        # 定时刷新 Tkinter 窗口
        self.create_timer(0.05, self.update_window)

        self.get_logger().info('Scene visualizer started.')

    def send_command(self):
        """把输入框中的自然语言任务发布到 ROS 2。"""
        command = self.command_entry.get().strip()

        if not command:
            self.status_label.config(
                text='Please enter a task command.',
                bg='#b71c1c'
            )
            return

        message = String()
        message.data = command
        self.command_publisher.publish(message)

        self.status_label.config(
            text=f'Task submitted: {command}',
            bg='#1565c0'
        )

        self.get_logger().info(
            f'Published user command: {command}'
        )

    def update_window(self):
        """在 ROS 2 循环中刷新 Tkinter 窗口。"""
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.get_logger().info('Visualizer window closed.')
            rclpy.shutdown()

    def draw_scene(self):
        """绘制当前二维场景。"""
        self.canvas.delete('all')

        # 标题
        self.canvas.create_text(
            450,
            35,
            text='Embodied Agent Task Execution',
            font=('Arial', 22, 'bold'),
            fill='#263238'
        )

        # 桌面
        self.canvas.create_rectangle(
            70,
            180,
            830,
            440,
            fill='#d7b98e',
            outline='#795548',
            width=4
        )

        self.canvas.create_text(
            110,
            205,
            text='TABLE',
            font=('Arial', 11, 'bold'),
            fill='#795548'
        )

        # 篮子
        self.canvas.create_rectangle(
            610,
            290,
            790,
            405,
            fill='#9b6a3c',
            outline='#5d4037',
            width=4
        )

        self.canvas.create_line(
            625,
            290,
            650,
            405,
            fill='#6d4c41',
            width=3
        )
        self.canvas.create_line(
            675,
            290,
            690,
            405,
            fill='#6d4c41',
            width=3
        )
        self.canvas.create_line(
            725,
            290,
            730,
            405,
            fill='#6d4c41',
            width=3
        )
        self.canvas.create_line(
            775,
            290,
            770,
            405,
            fill='#6d4c41',
            width=3
        )

        self.canvas.create_text(
            700,
            425,
            text='Basket',
            font=('Arial', 13, 'bold'),
            fill='#4e342e'
        )

        # 蓝色杯子
        self.canvas.create_rectangle(
            390,
            305,
            475,
            395,
            fill='#42a5f5',
            outline='#1565c0',
            width=4
        )
        self.canvas.create_oval(
            455,
            325,
            510,
            370,
            outline='#1565c0',
            width=5
        )
        self.canvas.create_text(
            430,
            420,
            text='Blue Cup',
            font=('Arial', 13, 'bold'),
            fill='#0d47a1'
        )


        # 苹果
        self.canvas.create_oval(
            self.apple_x - 35,
            self.apple_y - 35,
            self.apple_x + 35,
            self.apple_y + 35,
            fill='#e53935',
            outline='#8e0000',
            width=4
        )
        self.canvas.create_line(
            self.apple_x,
            self.apple_y - 35,
            self.apple_x + 5,
            self.apple_y - 52,
            fill='#4e342e',
            width=5
        )

        if self.apple_state != 'basket':
            self.canvas.create_text(
                self.apple_x,
                self.apple_y + 55,
                text='Red Apple',
                font=('Arial', 13, 'bold'),
                fill='#8e0000'
            )

        if self.apple_state == 'holding':
            self.canvas.create_text(
                self.apple_x,
                self.apple_y - 65,
                text='Robot is holding the apple',
                font=('Arial', 13, 'bold'),
                fill='#ef6c00'
            )

    def animate_apple(self, target_x, target_y, duration=1.5):
        """让苹果从当前位置平滑移动到目标位置。"""
        start_x = self.apple_x
        start_y = self.apple_y
        frames = 40

        for frame in range(1, frames + 1):
            progress = frame / frames

            self.apple_x = start_x + (target_x - start_x) * progress
            self.apple_y = start_y + (target_y - start_y) * progress

            self.draw_scene()
            self.root.update_idletasks()
            self.root.update()

            time.sleep(duration / frames)

    def execution_status_callback(self, msg):
        """根据执行状态更新画面。"""
        raw_status = msg.data
        status_text = raw_status.lower()

        # 同时兼容普通文本和 JSON 消息
        try:
            status_data = json.loads(raw_status)

            # 只分析当前执行步骤，避免 command 中的完整任务反复触发动作
            status_text = str(
                status_data.get(
                    'step',
                    status_data.get(
                        'failed_step',
                        status_data.get('reason', status_data.get('status', ''))
                    )
                )
            ).lower()

        except json.JSONDecodeError:
            status_data = {}
            status_text = raw_status.lower()

        self.get_logger().info(f'Received status: {raw_status}')
        self.status_label.config(text=raw_status)

        failure_words = ['failed', 'failure', 'error', '失败']
        is_failure = any(word in status_text for word in failure_words)

        # 失败时不改变物体位置
        if is_failure:
            self.status_label.config(
                text=f'Execution failed: {raw_status}',
                bg='#b71c1c'
            )
            return

        self.status_label.config(bg='#263238')

        # 成功抓取苹果
        pick_actions = ['pick', 'grasp', '抓取', '拾取', '夹取', '拿起']
        apple_words = ['apple', '苹果']

        if (
            any(word in status_text for word in pick_actions)
            and any(word in status_text for word in apple_words)
        ):
            self.apple_state = 'holding'
            self.animate_apple(530, 150)

        # 成功把苹果放入篮子
        place_actions = ['place', 'put', '放置', '放入', '置入']
        basket_words = ['basket', '篮子', '篮筐']

        if (
            any(word in status_text for word in place_actions)
            and any(word in status_text for word in apple_words)
            and any(word in status_text for word in basket_words)
        ):
            self.apple_state = 'basket'
            self.animate_apple(700, 340)


def main(args=None):
    rclpy.init(args=args)
    node = SceneVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()