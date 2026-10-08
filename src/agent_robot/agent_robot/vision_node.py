"""Backward-compatible entry point for structured OpenCV perception."""

from agent_robot.perception.perception_node import (
    PerceptionNode as VisionNode,
)
from agent_robot.perception.perception_node import main

__all__ = ['VisionNode', 'main']


if __name__ == '__main__':
    main()
