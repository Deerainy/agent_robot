"""Batch runner for M4 comparative experiments (current system).

Runs labeled suite tasks against fresh headless stacks (one launch per
task, zero world-state cross-talk). Two modes:

- ``online``: publish ``/user_command``; the live DeepSeek planner drives
  the executor; observed ``/task_plan`` payloads are cached per task;
- ``replay``: planner not started; cached plans are replayed to the
  executor in order, sending the next cached plan after each ``failed``
  status (faithfully reproducing the observed replan sequence).

Each task's recorder JSON is moved to ``<cond_dir>/<task_id>.json``;
shutdown-flushed ``incomplete`` runs are patched to ``failed`` when the
runner observed a terminal failure without replan.
"""

import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import time

import rclpy
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

from agent_robot.experiments.evaluation import default_suite_path, load_suite

SCENE_GRAPH_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

TERMINAL_STATUSES = ('succeeded', 'rejected')
REPLAN_GRACE_SECONDS = 25.0
RUN_DEADLINE_SECONDS = 150.0
STARTUP_WAIT_SECONDS = 12.0
SHUTDOWN_WAIT_SECONDS = 20.0

CONDITION_CURRENT = 'current'


class ReplaySequencer(object):
    """Pure state machine deciding which cached plan to send next."""

    def __init__(self, plans):
        # type: (list) -> None
        if not plans:
            raise ValueError('Replay cache must contain at least one plan.')

        self._plans = list(plans)
        self._sent = 0

    @property
    def total_plans(self):
        return len(self._plans)

    def initial_plan(self):
        # type: () -> dict
        self._sent = 1
        return self._plans[0]

    def plan_after_failure(self):
        # type: () -> (dict | None)
        if self._sent >= len(self._plans):
            return None

        plan = self._plans[self._sent]
        self._sent += 1
        return plan


class BatchIO(object):
    """Persistent subscriptions/publishers reused across task runs."""

    def __init__(self, node):
        self.node = node
        self.statuses = []
        self.plans = []
        self.graph_messages = []
        self.on_failure = None

        self.plan_publisher = node.create_publisher(
            String, '/task_plan', 10
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
            String, '/scene_graph', self._graph_callback, SCENE_GRAPH_QOS
        )

    def reset(self):
        self.statuses = []
        self.plans = []
        self.graph_messages = []
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

    def _graph_callback(self, msg):
        self.graph_messages.append(msg)


def observed_outcome(statuses):
    # type: (list) -> ((str | None), str)
    """Infer the final outcome from the observed status stream."""
    for status in reversed(statuses):
        if status.get('status') in TERMINAL_STATUSES:
            return status['status'], status.get('reason', '')

    failures = [
        status for status in statuses
        if status.get('status') == 'failed'
    ]

    if failures:
        return 'failed', failures[-1].get('reason', '')

    return None, ''


SUBSCRIPTION_MATCH_GRACE = 1.5


def wait_stack_ready(node, batch):
    # type: (object, BatchIO) -> None
    """Wait for the executor (scene graph) and the recorder to be up,
    then give ROS a short grace period so every subscription is matched
    before the first plan is published. Feasible=false plans are answered
    by the executor in milliseconds; without this window the recorder's
    /task_status subscription can miss the terminal rejected status.
    """
    deadline = time.time() + STARTUP_WAIT_SECONDS

    while time.time() < deadline and not batch.graph_messages:
        rclpy.spin_once(node, timeout_sec=0.2)

    deadline = time.time() + STARTUP_WAIT_SECONDS

    while time.time() < deadline:
        names = [
            name for name, _namespace in node.get_node_names_and_namespaces()
        ]

        if 'trajectory_recorder' in names and 'task_executor' in names:
            break

        rclpy.spin_once(node, timeout_sec=0.2)

    time.sleep(SUBSCRIPTION_MATCH_GRACE)


def spin_until_outcome(node, statuses):
    # type: (object, list) -> ((str | None), str)
    deadline = time.time() + RUN_DEADLINE_SECONDS
    last_activity = time.time()
    last_count = 0

    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        outcome, reason = observed_outcome(statuses)

        if outcome in TERMINAL_STATUSES:
            return outcome, reason

        if len(statuses) != last_count:
            last_count = len(statuses)
            last_activity = time.time()

        if outcome == 'failed' \
                and time.time() - last_activity > REPLAN_GRACE_SECONDS:
            return outcome, reason

    return observed_outcome(statuses)


def launch_stack(workspace, task_tmp_dir, mode):
    # type: (str, str, str) -> subprocess.Popen
    command = (
        'source /opt/ros/foxy/setup.bash && '
        'source {workspace}/install/setup.bash && '
        'export ROS_LOCALHOST_ONLY=1 && '
        'ros2 launch agent_robot experiment_minimal.launch.py '
        'backend:=mock use_planner:={planner} '
        'output_dir:={output_dir}'
    ).format(
        workspace=workspace,
        planner='false' if mode == 'replay' else 'true',
        output_dir=task_tmp_dir,
    )

    return subprocess.Popen(
        ['bash', '-c', command],
        cwd=workspace,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def _descendant_pids(pid):
    # type: (int) -> list
    """Return all descendant PIDs of *pid* by walking /proc."""
    children = []

    try:
        with open('/proc/{0}/task/{0}/children'.format(pid)) as handle:
            children = [int(x) for x in handle.read().split()]
    except (OSError, ValueError):
        return []

    descendants = list(children)

    for child in children:
        descendants.extend(_descendant_pids(child))

    return descendants


def stop_stack(process):
    # type: (subprocess.Popen) -> None
    """Stop a launched stack and every process it spawned.

    ``ros2 launch`` / ``ros2 run`` create nested process groups and may
    reparent node processes; the bash wrapper can exit before its
    grandchildren.  We therefore collect every descendant PID first,
    signal them all, and SIGKILL any survivors.
    """
    if process.poll() is not None:
        return

    targets = [process.pid] + _descendant_pids(process.pid)

    for pid in targets:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            pass

    try:
        process.wait(timeout=SHUTDOWN_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        pass

    survivors = [process.pid] + _descendant_pids(process.pid)

    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def collect_run_file(task_tmp_dir, condition_dir, task_id,
                     observed, observed_reason):
    # type: (str, str, str, str | None, str) -> (str | None)
    files = [
        path for path in glob.glob(os.path.join(task_tmp_dir, '*.json'))
    ]

    if not files:
        return None

    with open(files[0], encoding='utf-8') as handle:
        run = json.load(handle)

    if run.get('outcome') == 'incomplete' and observed == 'failed':
        # The recorder flushes failed-open runs as incomplete on SIGINT;
        # the runner's observed status stream is authoritative here.
        run['outcome'] = 'failed'
        run['outcome_reason'] = observed_reason or (
            'failed with no replan (batch runner shutdown)'
        )

    destination = os.path.join(
        condition_dir, '{}.json'.format(task_id)
    )

    with open(destination, 'w', encoding='utf-8') as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)

    os.remove(files[0])
    return destination


def run_one_task(batch, task, mode, workspace, condition_dir,
                 tmp_root, cache_dir):
    # type: (BatchIO, dict, str, str, str, str, str) -> dict
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

    process = launch_stack(workspace, task_tmp_dir, mode)

    try:
        wait_stack_ready(batch.node, batch)

        if mode == 'replay':
            batch.publish_plan(sequencer.initial_plan())
        else:
            batch.publish_command(command)

        outcome, reason = spin_until_outcome(batch.node, batch.statuses)

        # Let the recorder flush terminal runs (0.5s wait + margin).
        deadline = time.time() + 2.5

        while time.time() < deadline:
            rclpy.spin_once(batch.node, timeout_sec=0.1)
    finally:
        stop_stack(process)

    path = collect_run_file(
        task_tmp_dir, condition_dir, task_id, outcome, reason
    )

    if mode == 'online':
        with open(
            os.path.join(cache_dir, '{}.json'.format(task_id)),
            'w', encoding='utf-8'
        ) as handle:
            json.dump(
                {'task_id': task_id, 'plans': batch.plans},
                handle,
                ensure_ascii=False,
                indent=2
            )

    return {
        'task_id': task_id,
        'outcome': outcome,
        'reason': reason,
        'plans_observed': len(batch.plans),
        'run_file': path,
    }


def _replay_next(batch, sequencer):
    next_plan = sequencer.plan_after_failure()

    if next_plan is not None:
        batch.publish_plan(next_plan)


def parse_task_selection(value, suite):
    # type: (str | None, list) -> list
    if not value:
        return suite

    wanted = {item.strip() for item in value.split(',') if item.strip()}
    selected = [task for task in suite if task['id'] in wanted]

    if len(selected) != len(wanted):
        known = {task['id'] for task in suite}
        raise ValueError(
            'Unknown task ids: {}'.format(
                ', '.join(sorted(wanted - known))
            )
        )

    return selected


def write_manifest(condition_dir, mode, workspace, tasks, results):
    # type: (str, str, str, list, list) -> None
    git_describe = ''

    try:
        git_describe = subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            cwd=workspace,
            stderr=subprocess.DEVNULL,
        ).decode('utf-8').strip()
    except (subprocess.CalledProcessError, OSError):
        pass

    with open(default_suite_path(), 'rb') as handle:
        suite_hash = hashlib.sha256(handle.read()).hexdigest()

    manifest = {
        'condition': CONDITION_CURRENT,
        'mode': mode,
        'backend': 'mock',
        'git_describe': git_describe,
        'suite_sha256': suite_hash,
        'generated_at': time.time(),
        'task_ids': [task['id'] for task in tasks],
        'results': results,
    }

    with open(
        os.path.join(condition_dir, 'manifest.json'), 'w',
        encoding='utf-8'
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=('online', 'replay'), default='replay'
    )
    parser.add_argument(
        '--tasks',
        help='Comma-separated task ids; defaults to the whole suite.'
    )
    parser.add_argument('--suite', help='Suite JSON path.')
    parser.add_argument(
        '--workspace',
        default=os.path.expanduser('~/ros2_ws')
    )
    parser.add_argument(
        '--results-root',
        default=os.path.expanduser('~/ros2_ws/experiments')
    )
    arguments = parser.parse_args(args)

    suite = load_suite(arguments.suite)
    tasks = parse_task_selection(arguments.tasks, suite)

    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    result_dir = os.path.join(
        arguments.results_root, 'results', stamp, CONDITION_CURRENT
    )
    cache_dir = os.path.join(
        arguments.results_root, 'cache', CONDITION_CURRENT
    )
    tmp_root = os.path.join(
        arguments.results_root, 'results', stamp, '_tmp'
    )
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    rclpy.init()
    node = rclpy.create_node('m4_batch_runner')
    batch = BatchIO(node)
    results = []

    try:
        for task in tasks:
            print(
                '[{}] {} ...'.format(arguments.mode, task['id']),
                flush=True
            )
            result = run_one_task(
                batch,
                task,
                arguments.mode,
                arguments.workspace,
                result_dir,
                tmp_root,
                cache_dir,
            )
            results.append(result)
            print(
                '    -> {} (plans={}, file={})'.format(
                    result['outcome'],
                    result['plans_observed'],
                    'yes' if result['run_file'] else 'MISSING',
                ),
                flush=True
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    write_manifest(
        result_dir, arguments.mode, arguments.workspace, tasks, results
    )
    print('Wrote manifest to {}'.format(result_dir), flush=True)


if __name__ == '__main__':
    main()
