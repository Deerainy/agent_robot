from agent_robot.web_demo import EpisodeCollector, _render_summary


def test_web_summary_shows_execution_and_recovery_fields():
    summary = _render_summary({
        'detected_objects': [{'id': 'red_apple_1'}],
        'plan': {
            'actions': [
                {'skill': 'pick', 'object': 'red_apple_1'},
            ],
        },
        'execution_status': {
            'status': 'succeeded',
            'reason': 'All actions completed.',
        },
        'trajectory_point_count': 4,
        'failures': [{
            'failure_code': 'grasp_failed',
            'reason': 'Injected grasp failure.',
        }],
        'replans': 1,
        'recovered': True,
    })

    assert '**succeeded**' in summary
    assert 'red_apple_1' in summary
    assert '4' in summary
    assert '恢复成功：是' in summary
    assert 'grasp_failed' in summary
    assert '"skill": "pick"' in summary


def test_collector_waits_for_retry_after_grasp_failure():
    collector = object.__new__(EpisodeCollector)
    collector.statuses = [{
        'status': 'failed',
        'failure_code': 'grasp_failed',
    }]
    collector.plans = []

    assert not collector.is_finished(
        failure_started=None,
        recovery_wait=12.0,
    )
    collector.plans = [{'status': 'replanned'}]
    assert not collector.is_finished(
        failure_started=None,
        recovery_wait=12.0,
    )


def test_collector_finishes_on_terminal_status_or_non_grasp_failure():
    collector = object.__new__(EpisodeCollector)
    collector.plans = []
    collector.statuses = [{'status': 'succeeded'}]
    assert collector.is_finished(None, 12.0)

    collector.statuses = [{
        'status': 'failed',
        'failure_code': 'skill_backend_unavailable',
    }]
    assert collector.is_finished(None, 12.0)


def test_web_error_summary_is_visible():
    assert '任务未完成' in _render_summary({'error': 'ROS timeout'})
    assert 'ROS timeout' in _render_summary({'error': 'ROS timeout'})
