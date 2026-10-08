"""Execution trajectory assembly for EmbodiedPlan (M3).

Pure logic, free of ROS imports: assembles at most one active Run at a
time (single robot) from the plan/status/scene-graph message streams, so
the recorder node and the offline tests share one implementation.

Run lifecycle:

- the first plan/status event opens the run;
- ``succeeded`` / ``rejected`` are terminal: the run becomes ready for
  flushing after ``FINAL_GRAPH_WAIT`` seconds so the executor's final
  scene graph publish lands in ``scene_graph_final``;
- ``failed`` is not terminal: the run stays open for a replan; if no
  event arrives within ``IDLE_FLUSH_TIMEOUT`` the run is flushed as
  ``failed`` (or ``incomplete`` when nothing but a plan was seen);
- an event for a different command while a run is active closes the
  active run by the same rules and opens a new one (defensive: the
  executor processes one plan at a time).
"""

import time
import uuid

RUN_SCHEMA_VERSION = '1.0'

# Wait after a terminal status for the executor's final scene graph.
FINAL_GRAPH_WAIT = 0.5

# Flush an open run that has seen no events for this long. A run with a
# failed event and no replan flushes as ``failed``; otherwise as
# ``incomplete``.
IDLE_FLUSH_TIMEOUT = 120.0

OUTCOME_SUCCEEDED = 'succeeded'
OUTCOME_FAILED = 'failed'
OUTCOME_REJECTED = 'rejected'
OUTCOME_INCOMPLETE = 'incomplete'

# Message fields copied verbatim into run records.
_PLAN_FIELDS = (
    'feasible',
    'failure_reason',
    'actions',
    'planner',
    'replans',
    'reason',
    'based_on_world_revision',
    'image_path',
    'status',
)
_STATUS_FIELDS = (
    'status',
    'action',
    'step_index',
    'total_steps',
    'reason',
    'world_revision',
    'failure_code',
)


def make_run_id(clock):
    # type: (object) -> str
    """``run_<UTC timestamp>_<uuid8>``: sortable prefix, unique suffix."""
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime(clock()))
    return 'run_{}_{}'.format(stamp, uuid.uuid4().hex[:8])


class RunBuilder(object):
    """Assemble at most one active Run at a time (single robot)."""

    def __init__(self, clock=time.time, monotonic=time.monotonic):
        # type: (object, object) -> None
        self._clock = clock
        self._mono = monotonic
        self._active = None  # type: (dict | None)
        self._finished = []  # type: (list)
        self._latest_graph = None  # type: (dict | None)
        self._scenario = {}  # type: dict
        self._latest_world_revision = None  # type: (int | None)
        self._current_scenario_id = None  # type: (str | None)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_plan(self, plan):
        # type: (dict) -> None
        if not isinstance(plan, dict):
            return

        command = self._command_of(plan)
        self._ensure_run_for(command)

        self._active['plans'].append(
            {field: plan.get(field) for field in _PLAN_FIELDS}
        )
        self._active['_last_event_mono'] = self._mono()

    def on_status(self, status):
        # type: (dict) -> None
        if not isinstance(status, dict):
            return

        command = self._command_of(status)
        self._ensure_run_for(command)

        run = self._active
        event = {'t': self._clock(), 'mono': self._mono()}
        event.update({field: status.get(field) for field in _STATUS_FIELDS})
        run['events'].append(event)
        run['_last_event_mono'] = event['mono']

        name = status.get('status')
        if name in ('action_started', 'action_completed'):
            action = status.get('action') or {}
            params = action.get('params') or {}
            run['action_revisions'].append({
                'status': name,
                'step_index': status.get('step_index'),
                'skill': action.get('skill'),
                'object': params.get('object', action.get('object')),
                'target': params.get('target', action.get('target')),
                'world_revision': status.get('world_revision'),
            })
        elif name == 'failed':
            reason = status.get('reason', '')
            if reason:
                run['failure_reasons'].append(reason)
            run['failures'].append({
                'failure_code': status.get('failure_code'),
                'action': status.get('action'),
                'step_index': status.get('step_index'),
                'reason': reason,
            })
            run['_has_failure'] = True

        if name == 'succeeded':
            run['_outcome'] = OUTCOME_SUCCEEDED
            run['_outcome_reason'] = status.get('reason', '')
            run['_close_at_mono'] = event['mono'] + FINAL_GRAPH_WAIT
        elif name == 'rejected':
            run['_outcome'] = OUTCOME_REJECTED
            run['_outcome_reason'] = status.get('reason', '')
            run['_close_at_mono'] = event['mono'] + FINAL_GRAPH_WAIT
        elif name == 'failed':
            run['_has_failure'] = True

    def on_scene_graph(self, graph):
        # type: (dict) -> None
        """Cache the latest graph; refresh the active run's final view."""
        if not isinstance(graph, dict):
            return

        self._latest_graph = graph

        if self._active is not None:
            self._active['scene_graph_final'] = graph

    def on_environment(self, environment):
        # type: (dict) -> None
        """Track episode identity and monotonically increasing revisions."""
        if not isinstance(environment, dict):
            return

        scenario_id = environment.get('scenario_id')
        if (
            isinstance(scenario_id, str)
            and self._current_scenario_id is not None
            and scenario_id != self._current_scenario_id
        ):
            self._latest_world_revision = None
        if isinstance(scenario_id, str):
            self._current_scenario_id = scenario_id

        for field in (
            'scenario_id', 'category', 'seed', 'image_path', 'source'
        ):
            if field in environment:
                self._scenario[field] = environment[field]
        if self._active is not None:
            self._active['scenario'].update(self._scenario)

        revision = environment.get('world_revision')
        if not isinstance(revision, int):
            return
        if (
            self._latest_world_revision is not None
            and revision <= self._latest_world_revision
        ):
            return

        previous = self._latest_world_revision
        self._latest_world_revision = revision

        if self._active is not None:
            self._active['world_revisions'].append(revision)
            if previous is not None:
                self._active['world_events'].append({
                    'kind': 'revision_changed',
                    'from_revision': previous,
                    'revision': revision,
                })

    def on_world_event(self, event):
        # type: (dict) -> None
        """Record an explicit physics-backend world event."""
        if isinstance(event, dict) and self._active is not None:
            self._active['world_events'].append(dict(event))

    def on_trajectory_point(self, payload):
        # type: (dict) -> None
        if not isinstance(payload, dict) or self._active is None:
            return
        self._active['trajectory'].append(dict(payload))
        self._active['_last_event_mono'] = self._mono()

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def pop_ready_runs(self):
        # type: () -> list
        """Return all runs due for flushing and stop tracking them."""
        ready = list(self._finished)
        self._finished = []

        now = self._mono()

        if self._active is not None:
            run = self._active

            close_at = run.get('_close_at_mono')

            if close_at is not None and now >= close_at:
                ready.append(self._finalize(run))
                self._active = None
            elif now - run['_last_event_mono'] >= IDLE_FLUSH_TIMEOUT:
                ready.append(self._finalize(run))
                self._active = None

        return ready

    def close_active(self, reason):
        # type: (str) -> dict | None
        """Force-close the active run (shutdown or superseded command)."""
        if self._active is None:
            return None

        run = self._finalize(self._active, force_reason=reason)
        self._active = None
        return run

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_run_for(self, command):
        # type: (str) -> None
        if self._active is not None:
            if self._active['command'] == command:
                return
            superseded = self.close_active('superseded_by_new_command')

            if superseded is not None:
                self._finished.append(superseded)

        now = self._clock()
        now_mono = self._mono()
        self._active = {
            'schema_version': RUN_SCHEMA_VERSION,
            'run_id': make_run_id(self._clock),
            'command': command,
            'started_at': now,
            'finished_at': None,
            'started_mono': now_mono,
            'finished_mono': None,
            'plans': [],
            'trajectory': [],
            'events': [],
            'outcome': None,
            'outcome_reason': '',
            'scenario': dict(self._scenario),
            'world_revisions': (
                [self._latest_world_revision]
                if self._latest_world_revision is not None else []
            ),
            'action_revisions': [],
            'world_events': [],
            'failure_reasons': [],
            'failures': [],
            # State just before the run opened (may be the startup graph
            # latched long before this run existed).
            'scene_graph_initial': self._latest_graph,
            'scene_graph_final': self._latest_graph,
            '_last_event_mono': now_mono,
            '_has_failure': False,
        }

    def _finalize(self, run, force_reason=None):
        # type: (dict, str | None) -> dict
        outcome = run.pop('_outcome', None)

        if outcome is None:
            if run['_has_failure']:
                outcome = OUTCOME_FAILED
                default_reason = (
                    'no replan within {}s after the last failure'.format(
                        IDLE_FLUSH_TIMEOUT
                    )
                )
            else:
                outcome = OUTCOME_INCOMPLETE
                default_reason = 'flushed without a terminal status'
        else:
            default_reason = run.pop('_outcome_reason', '')

        run['outcome'] = outcome
        run['outcome_reason'] = force_reason or default_reason or ''
        run['image'] = run['scenario'].get('image_path', '')
        run['instruction'] = run['command']
        run['plan'] = list(run['plans'])
        run['success'] = outcome == OUTCOME_SUCCEEDED
        run['replans'] = max(
            [
                plan.get('replans', 0)
                for plan in run['plans']
                if isinstance(plan.get('replans'), int)
            ] or [0]
        )
        run['recovered'] = bool(
            run['success'] and run['failures'] and run['replans'] > 0
        )
        run['finished_at'] = self._clock()
        run['finished_mono'] = self._mono()

        for key in ('_close_at_mono', '_last_event_mono', '_has_failure'):
            run.pop(key, None)

        return run

    @staticmethod
    def _command_of(message):
        # type: (dict) -> str
        if not isinstance(message, dict):
            return ''

        command = message.get('command')

        return command if isinstance(command, str) else ''
