import base64
import json
import os
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        self.image_subscription = self.create_subscription(
            String,
            '/image_path',
            self.image_callback,
            10
        )

        self.environment_publisher = self.create_publisher(
            String,
            '/environment_state',
            10
        )

        self.get_logger().info(
            'Vision node has started. Waiting for /image_path.'
        )

    def image_callback(self, msg):
        image_path = os.path.abspath(
            os.path.expanduser(msg.data.strip())
        )

        if not os.path.isfile(image_path):
            self.get_logger().error(
                f'Image not found: {image_path}'
            )
            return

        api_key = os.environ.get('DEEPSEEK_API_KEY')

        if not api_key:
            self.get_logger().error(
                'DEEPSEEK_API_KEY is not configured.'
            )
            return

        self.get_logger().info(
            f'Analyzing image: {image_path}'
        )

        try:
            environment = self.analyze_image(
                image_path,
                api_key
            )

            message = String()
            message.data = json.dumps(
                environment,
                ensure_ascii=False
            )

            self.environment_publisher.publish(message)

            self.get_logger().info(
                f'Published visual environment: {message.data}'
            )

        except urllib.error.HTTPError as error:
            error_body = error.read().decode('utf-8')
            self.get_logger().error(
                f'Vision HTTP error {error.code}: {error_body}'
            )

        except Exception as error:
            self.get_logger().error(
                f'Vision analysis failed: {error}'
            )

    def analyze_image(self, image_path, api_key):
        suffix = os.path.splitext(image_path)[1].lower()

        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }

        mime_type = mime_types.get(suffix)

        if mime_type is None:
            raise ValueError(
                'Only JPG, JPEG, PNG, WEBP and GIF are supported.'
            )

        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(
                image_file.read()
            ).decode('utf-8')

        image_url = (
            f'data:{mime_type};base64,{encoded_image}'
        )

        prompt = """
分析这张机器人视角的场景图片，识别适合机器人操作的主要物体。

要求：
1. 识别物体名称、颜色和大致位置。
2. 位置使用简短描述，例如table、floor、left或right。
3. 不要识别背景中无关的小物体。
4. 不确定的信息不要虚构。
5. 只输出JSON。

格式：

{
    "robot_location": "unknown",
    "objects": [
        {
            "name": "物体英文名称",
            "color": "颜色",
            "location": "位置"
        }
    ]
}
"""

        request_data = {
            'model': 'deepseek-flash',
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': image_url,
                                'detail': 'low'
                            }
                        }
                    ]
                }
            ],
            'response_format': {
                'type': 'json_object'
            },
            'thinking': {
                'type': 'disabled'
            },
            'stream': False
        }

        request = urllib.request.Request(
            'https://api.deepseek.com/chat/completions',
            data=json.dumps(request_data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            method='POST'
        )

        with urllib.request.urlopen(
            request,
            timeout=90
        ) as response:
            result = json.loads(
                response.read().decode('utf-8')
            )

        content = result['choices'][0]['message']['content']
        environment = json.loads(content)

        if not isinstance(environment.get('objects'), list):
            raise ValueError('Invalid visual environment output.')

        return environment


def main(args=None):
    rclpy.init(args=args)

    node = VisionNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()