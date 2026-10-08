"""
M5 multi-scene subsystem: specs, generators, goals, world model.

Every module in this package is pure logic: no ``rclpy`` or ``pybullet``
imports. ROS nodes and physics backends consume these structures but the
structures never depend on them.
"""
