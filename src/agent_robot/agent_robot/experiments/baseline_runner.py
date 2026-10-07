"""Batch runner for the v0.4 baseline condition.

Mirrors :mod:`agent_robot.experiments.batch_runner` but targets the v0.4
stack (environment_node + task_planner + task_executor, no scene graph).
The runner observes v0.4 plan/status streams and synthesises M3 run JSON
in-process — no separate adapter node.

Key differences from the current-system runner:

* Plans carry a free-text ``steps`` list instead of structured ``actions``;
* v0.4 has no ``succeeded`` status — the last ``completed`` step
  (``step_index == total_steps``) implies success;
* there is no scene graph, so final-state checks return ``None``.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid

import rclpy
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.experiments.batch_runner import (
    REPLAN_GRACE_SECONDS,
    RUN_DEADLINE_SECONDS,
    STARTUP_WAIT_SECONDS,
    ReplaySequencer,
    stop_stack,
)
from agent_robot.experiments.evaluation import default_suite_path, load_suite

PLAN_QOS = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

TERMINAL_STATUSES = ('succeeded', 'rejected')
CONDITION_BASELINE = 'baseline'


class BaselineIO(object):
    """Subscriptions/publishers for the v0.4 stack (no scene graph)."""

    def __init__(self, node):
        self.node = node
        self.statuses = []
        self.plans = []
        self.environment_messages = []
        self.on_failure = None

        self.plan_publisher = node.create_publisher(
            String, '/task_plan', PLAN_QOS
        )
        self.command_publisher = node.create_publisher(
            String, '/user_command', 10
        )
        node.create_subscription(
            String, '/task_status', self._status_callback, 10
        )
        node.create_subscription(
            String, '/task_plan', self._plan_callback, 10
        )
        node.create_subscription(
            String, '/environment_state', self._env_callback, 10
        )

    def reset(self):
        self.statuses = []
        self.plans = []
        self.environment_messages = []
        self.on_failure = None

    def publish_plan(self, plan):
        message = String()
        message.data = json.dumps(plan, ensure_ascii=False)
        self.plan_publisher.publish(message)

    def publish_command(self, command):
        message = String()
        message.data = command
        self.command_publisher.publish(message)

    def _status_callback(self, msg):
        payload = json.loads(msg.data)
        self.statuses.append(payload)

        if payload.get('status') == 'failed' and self.on_failure:
            self.on_failure()

    def _plan_callback(self, msg):
        self.plans.append(json.loads(msg.data))

    def _env_callback(self, msg):
        self.environment_messages.append(msg)


def wait_baseline_ready(node, batch):
    # type: (object, BaselineIO) -> None
    """Wait for environment_node (executor up) and the adapter, then a
    short grace period so all subscriptions are matched before the first
    plan or command is published.
    """
    deadline = time.time() + STARTUP_WAIT_SECONDS

    while time.time() < deadline and not batch.environment_messages:
        rclpy.spin_once(node, timeout_sec=0.2)

    deadline = time.time() + STARTUP_WAIT_SECONDS

    while time.time() < deadline:
        names = [
            name for name, _namespace in node.get_node_names_and_namespaces()
        ]

        if 'baseline_adapter' in names and 'task_executor' in names:
            break

        rclpy.spin_once(node, timeout_sec=0.2)

    time.sleep(5.0)


def baseline_outcome(statuses):
    # type: (list) -> ((str | None), str)
    """Infer the v0.4 terminal outcome from the status stream.

    v0.4 never emits ``succeeded``; a run succeeds when the final
    ``completed`` status reports ``step_index == total_steps``.
    """
    for status in reversed(statuses):
        if status.get('status') == 'rejected':
            return 'rejected', status.get('reason', '')

        if status.get('status') == 'completed':
            if status.get('step_index', 0) >= status.get(
                'total_steps', 0
            ):
                return 'succeeded', ''

    failures = [
        s for s in statuses if s.get('status') == 'failed'
    ]

    if failures:
        return 'failed', failures[-1].get('reason', '')

    return None, ''


def spin_until_baseline_outcome(node, statuses):
    # type: (object, list) -> ((str | None), str)
    deadline = time.time() + RUN_DEADLINE_SECONDS
    last_activity = time.time()
    last_count = 0

    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        outcome, reason = baseline_outcome(statuses)

        if outcome in TERMINAL_STATUSES:
            return outcome, reason

        if len(statuses) != last_count:
            last_count = len(statuses)
            last_activity = time.time()

        if outcome == 'failed' \
                and time.time() - last_activity > REPLAN_GRACE_SECONDS:
            return outcome, reason

    return baseline_outcome(statuses)


def launch_baseline_stack(v04_workspace, task_tmp_dir, mode):
    # type: (str, str, str) -> subprocess.Popen
    """Launch the v0.4 stack only.

    The run JSON is synthesised in-process by :func:`build_baseline_run`
    from the statuses the runner itself observes — no separate adapter
    node is needed, eliminating subscription-matching races.
    """
    command = (
        'source /opt/ros/foxy/setup.bash && '
        'source {v04}/install/setup.bash && '
        'export ROS_LOCALHOST_ONLY=1 && '
        'ros2 launch agent_robot baseline_minimal.launch.py'
    ).format(v04=v04_workspace)

    return subprocess.Popen(
        ['bash', '-c', command],
        cwd=v04_workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def build_baseline_run(command, plans, statuses, outcome, reason):
    # type: (str, list, list, str | None, str) -> dict
    """Translate observed v0.4 plan/status streams into an M3 run JSON.

    Each ``completed`` step expands to a synthetic ``action_started`` +
    ``action_completed`` pair; the final step appends ``succeeded``.
    The v0.4 stack has no scene graph, so graph snapshots are ``None``.
    """
    run_id = '{0}-{1}'.format(
        int(time.time() * 1000), uuid.uuid4().hex[:8]
    )

    normalised_plans = []

    for plan in plans:
        normalised_plans.append({
            'command': plan.get('command', command),
            'schema_version': '2.1',
            'feasible': plan.get('feasible', True),
            'actions': [
                {'skill': 'step', 'description': s}
                for s in plan.get('steps', [])
            ],
            'planner': plan.get('planner', 'unknown'),
            'reason': plan.get('reason', ''),
            'replans': len(normalised_plans),
        })

    events = []

    for status in statuses:
        kind = status.get('status')

        if kind == 'completed':
            step_index = status.get('step_index', 0)
            total_steps = status.get('total_steps', 0)
            step_text = status.get('step', '')

            events.append({
                'event_id': str(uuid.uuid4()),
                'status': 'action_started',
                'step': step_index,
                'description': step_text,
            })
            events.append({
                'event_id': str(uuid.uuid4()),
                'status': 'action_completed',
                'step': step_index,
                'description': step_text,
            })

            if step_index >= total_steps:
                events.append({
                    'event_id': str(uuid.uuid4()),
                    'status': 'succeeded',
                })

        elif kind == 'failed':
            events.append({
                'event_id': str(uuid.uuid4()),
                'status': 'failed',
                'reason': status.get('reason', ''),
                'failed_step': status.get('failed_step', ''),
            })

        elif kind == 'rejected':
            events.append({
                'event_id': str(uuid.uuid4()),
                'status': 'rejected',
                'reason': status.get('reason', ''),
            })

    return {
        'run_id': run_id,
        'command': command,
        'planner': normalised_plans[-1]['planner'] if normalised_plans
        else 'unknown',
        'schema_version': '2.1',
        'plans': normalised_plans,
        'events': events,
        'scene_graph_final': None,
        'outcome': outcome or 'incomplete',
        'outcome_reason': reason,
        'started_at_mono': 0.0,
        'duration_seconds': 0.0,
    }


def collect_run_file(task_tmp_dir, condition_dir, task_id, run):
    # type: (str, str, str, dict) -> (str | None)
    """Persist a synthesised baseline run to the condition directory."""
    destination = os.path.join(condition_dir, '{}.json'.format(task_id))

    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)

    return destination


def _replay_next(batch, sequencer):
    # type: (BaselineIO, ReplaySequencer) -> None
    plan = sequencer.plan_after_failure()

    if plan is not None:
        batch.publish_plan(plan)


def run_one_task(batch, task, mode, v04_workspace,
                 condition_dir, tmp_root, cache_dir):
    # type: (...) -> dict
    task_id = task['id']
    command = task['command']
    task_tmp_dir = os.path.join(tmp_root, task_id)
    os.makedirs(task_tmp_dir, exist_ok=True)

    batch.reset()
    sequencer = None

    if mode == 'replay':
        with open(
            os.path.join(cache_dir, '{}.json'.format(task_id)),
            encoding='utf-8'
        ) as handle:
            sequencer = ReplaySequencer(json.load(handle)['plans'])

        batch.on_failure = lambda: _replay_next(batch, sequencer)

    process = launch_baseline_stack(v04_workspace, task_tmp_dir, mode)

    try:
        wait_baseline_ready(batch.node, batch)

        if mode == 'replay':
            batch.publish_plan(sequencer.initial_plan())
        else:
            batch.publish_command(command)

        outcome, reason = spin_until_baseline_outcome(
            batch.node, batch.statuses
        )

        time.sleep(0.5)

    finally:
        stop_stack(process)

    run = build_baseline_run(
        command, batch.plans, batch.statuses, outcome, reason
    )
    collect_run_file(task_tmp_dir, condition_dir, task_id, run)

    return {
        'task_id': task_id,
        'outcome': outcome,
        'reason': reason,
        'plans': len(batch.plans),
        'file_path': '{}.json'.format(task_id),
    }


def parse_task_selection(value):
    # type: (str) -> list
    if not value or value.strip() == 'all':
        return []

    return [item.strip() for item in value.split(',') if item.strip()]


def suite_sha256(suite_path):
    with open(suite_path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_manifest(condition_dir, mode, task_ids, results, suite_path):
    try:
        version = subprocess.check_output(
            ['git', 'describe', '--tags', '--always'],
            cwd='/home/deerainy/ros2_ws',
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        version = 'unknown'

    manifest = {
        'condition': CONDITION_BASELINE,
        'mode': mode,
        'v04_workspace': '/tmp/embodiedplan_v04',
        'version': version,
        'suite_sha256': suite_sha256(suite_path),
        'task_ids': task_ids,
        'results': results,
    }

    with open(
        os.path.join(condition_dir, 'manifest.json'), 'w', encoding='utf-8'
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Run the M4 baseline (v0.4) experiment condition.'
    )
    parser.add_argument(
        '--mode', choices=['online', 'replay'], required=True
    )
    parser.add_argument(
        '--tasks', default='all',
        help='Comma-separated task ids, or "all".'
    )
    parser.add_argument(
        '--suite', default=default_suite_path(),
        help='Path to task_suite.json.'
    )
    parser.add_argument(
        '--workspace', default='/home/deerainy/ros2_ws',
        help='Current-system workspace (for the adapter entry point).'
    )
    parser.add_argument(
        '--v04-workspace', default='/tmp/embodiedplan_v04',
        help='v0.4 worktree path.'
    )
    parser.add_argument(
        '--results-root',
        default='/home/deerainy/ros2_ws/experiments/results',
    )

    cli = parser.parse_args(args)

    suite = load_suite(cli.suite)
    selection = parse_task_selection(cli.tasks)
    tasks = [
        task for task in suite if (not selection or task['id'] in selection)
    ]

    timestamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    base = os.path.join(cli.results_root, timestamp)
    condition_dir = os.path.join(base, CONDITION_BASELINE)
    tmp_root = os.path.join(base, '_tmp')
    os.makedirs(condition_dir, exist_ok=True)

    cache_dir = os.path.join(
        cli.workspace, 'experiments', 'cache', CONDITION_BASELINE
    )

    rclpy.init()
    node = rclpy.create_node('baseline_batch_runner')
    batch = BaselineIO(node)

    results = []

    try:
        for task in tasks:
            print('[baseline] {id} ...'.format(**task))
            result = run_one_task(
                batch, task, cli.mode,
                cli.v04_workspace,
                condition_dir, tmp_root, cache_dir,
            )
            results.append(result)
            file_flag = 'yes' if result['file_path'] else 'MISSING'
            print(
                '    -> {outcome} (plans={plans}, file={flag})'.format(
                    outcome=result['outcome'],
                    plans=result['plans'],
                    flag=file_flag,
                )
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    write_manifest(
        condition_dir, cli.mode,
        [t['id'] for t in tasks], results, cli.suite,
    )
    print('Wrote manifest to {}'.format(condition_dir))


if __name__ == '__main__':
    main()
