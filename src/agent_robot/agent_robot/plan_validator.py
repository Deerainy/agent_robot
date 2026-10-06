"""Pure accept/repair decision logic for LLM-generated plans (M2).

This module bridges the structured skill protocol, object grounding and the
scene-graph precondition validator. It imports neither rclpy nor pybullet, so
the planner node and the offline tests share one decision function.

Decision pipeline for a model JSON document:

1. schema sanity (``feasible`` flag, ``actions`` shape);
2. :func:`agent_robot.skill_registry.parse_actions`;
3. :func:`agent_robot.scene_graph.ground_action` against perceived names;
4. :func:`agent_robot.scene_graph.validate_actions` against the current
   scene graph (skipped when the caller has no graph yet).

Any failure is reported with a stage tag (``schema`` / ``grounding`` /
``validation``) so the planner can ask the model to self-repair exactly once.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_robot.scene_graph import (
    GroundingError,
    SceneGraph,
    ground_action,
    validate_actions,
)
from agent_robot.skill_registry import (
    ActionSchemaError,
    parse_actions,
)

STAGE_SCHEMA = 'schema'
STAGE_GROUNDING = 'grounding'
STAGE_VALIDATION = 'validation'


@dataclass
class PlanEvaluation:
    """Result of evaluating one raw model-generated plan document."""

    feasible: bool
    reason: str = ''
    actions: List[Dict[str, str]] = field(default_factory=list)
    status: str = 'rejected'
    error_stage: Optional[str] = None
    error_code: str = ''
    error_detail: str = ''
    violation_index: Optional[int] = None
    violation_action: Optional[Dict[str, str]] = None

    @property
    def accepted(self):
        # type: () -> bool
        return self.feasible and self.error_stage is None

    @property
    def repairable(self):
        # type: () -> bool
        return self.error_stage is not None


def evaluate_generated_plan(generated, env_names, graph):
    # type: (Any, List[str], Optional[SceneGraph]) -> PlanEvaluation
    """Validate one parsed model JSON document.

    ``env_names`` are the raw perceived object names (grounding matches them
    before canonical self-fallback). ``graph`` is the authoritative scene
    graph when available; ``None`` skips sequence validation (execution-time
    validation remains the safety net) and must never be faked.
    """
    if not isinstance(generated, dict):
        return PlanEvaluation(
            feasible=False,
            error_stage=STAGE_SCHEMA,
            error_detail='Model output is not a JSON object.'
        )

    feasible = generated.get('feasible', False)

    if not isinstance(feasible, bool):
        return PlanEvaluation(
            feasible=False,
            error_stage=STAGE_SCHEMA,
            error_detail='"feasible" must be a boolean.'
        )

    reason = generated.get('reason', '')

    if not isinstance(reason, str):
        reason = str(reason)

    raw_actions = generated.get('actions', [])

    if not isinstance(raw_actions, list):
        return PlanEvaluation(
            feasible=False,
            error_stage=STAGE_SCHEMA,
            error_detail='"actions" must be a list.'
        )

    if not feasible:
        if raw_actions:
            return PlanEvaluation(
                feasible=False,
                reason=reason,
                error_stage=STAGE_SCHEMA,
                error_detail='Infeasible plans must contain empty actions.'
            )

        return PlanEvaluation(
            feasible=False,
            reason=reason,
            status='rejected'
        )

    try:
        parsed_actions = parse_actions(raw_actions)
    except ActionSchemaError as error:
        return PlanEvaluation(
            feasible=True,
            reason=reason,
            status='planned',
            error_stage=STAGE_SCHEMA,
            error_detail=str(error)
        )

    grounded_actions = []

    for index, action in enumerate(parsed_actions):
        try:
            grounded_actions.append(ground_action(action, env_names))
        except GroundingError as error:
            return PlanEvaluation(
                feasible=True,
                reason=reason,
                actions=[a.to_dict() for a in grounded_actions],
                status='planned',
                error_stage=STAGE_GROUNDING,
                error_detail=str(error),
                violation_index=index,
                violation_action=action.to_dict()
            )

    if graph is not None:
        report = validate_actions(grounded_actions, graph)

        if not report.ok:
            violation = report.violation
            return PlanEvaluation(
                feasible=True,
                reason=reason,
                actions=[a.to_dict() for a in grounded_actions],
                status='planned',
                error_stage=STAGE_VALIDATION,
                error_code=violation.reason_code,
                error_detail=violation.detail,
                violation_index=violation.index,
                violation_action=violation.action.to_dict()
            )

    return PlanEvaluation(
        feasible=True,
        reason=reason,
        actions=[a.to_dict() for a in grounded_actions],
        status='planned'
    )


_STAGE_LABELS = {
    STAGE_SCHEMA: '动作协议不合法',
    STAGE_GROUNDING: '动作物体无法在当前场景中找到',
    STAGE_VALIDATION: '动作序列违反前置条件',
}


def build_validation_feedback(evaluation, graph):
    # type: (PlanEvaluation, SceneGraph) -> str
    """Assemble the Chinese prompt section for the one-shot self-repair."""
    lines = ['上一版计划未通过自动验证，必须修正后重新输出完整计划：']

    if evaluation.violation_index is not None:
        lines.append(
            '- 违例动作位置：actions[{}] = {}'.format(
                evaluation.violation_index,
                evaluation.violation_action
            )
        )

    stage_label = _STAGE_LABELS.get(
        evaluation.error_stage, evaluation.error_stage
    )
    lines.append(
        '- 问题类型：{}（{}）'.format(
            stage_label, evaluation.error_code or 'schema_error'
        )
    )
    lines.append('- 具体原因：{}'.format(evaluation.error_detail))

    if graph is not None:
        lines.append(graph.to_prompt_text())

    lines.append(
        '请针对上述原因修改动作序列（例如先 move_to 再 pick/place、'
        '更换为可抓取物体或容器目标），严禁原样重试；'
        '仍然只输出符合协议的 JSON。'
    )

    return '\n'.join(lines)
