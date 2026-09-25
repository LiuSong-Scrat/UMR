from unittest.mock import Mock

from rlbench.backend.waypoints import Point


def test_point_path_uses_waypoint_planner_timeout(monkeypatch):
    monkeypatch.setenv("RLBENCH_WAYPOINT_PLANNER_MAX_TIME_MS", "50")
    monkeypatch.setenv("RLBENCH_PLANNER_MAX_TIME_MS", "10")

    waypoint = Mock()
    waypoint.get_extension_string.return_value = ""
    robot = Mock()
    robot.arm.get_path.return_value = object()

    path = Point(waypoint, robot).get_path()

    assert path is robot.arm.get_path.return_value
    kwargs = robot.arm.get_path.call_args.kwargs
    assert kwargs["max_time_ms"] == 50
    assert kwargs["algorithm"].name == "RRTConnect"


def test_point_path_falls_back_to_general_planner_timeout(monkeypatch):
    monkeypatch.delenv("RLBENCH_WAYPOINT_PLANNER_MAX_TIME_MS", raising=False)
    monkeypatch.setenv("RLBENCH_PLANNER_MAX_TIME_MS", "35")

    waypoint = Mock()
    waypoint.get_extension_string.return_value = ""
    robot = Mock()
    robot.arm.get_path.return_value = object()

    Point(waypoint, robot).get_path()

    assert robot.arm.get_path.call_args.kwargs["max_time_ms"] == 35
