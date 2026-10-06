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
_PLAN_FIELDS = ('feasible', 'actions', 'planner', 'replans', 'reason')
_STATUS_FIELDS = ('status', 'action', 'step_index', 'total_steps', 'reason')


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
            'events': [],
            'outcome': None,
            'outcome_reason': '',
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
