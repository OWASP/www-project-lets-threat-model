import json
import uuid
import pytest
from core.agents.mitre_attack_agent import (
    MitreAttackAgent,
    AttackGraphStateModel,
    Result,
)
from langchain_core.language_models.chat_models import BaseChatModel
from core.models.dtos.MitreAttack import AgentAttack
from langchain_core.runnables import Runnable


def _extract_component(item) -> dict:
    if isinstance(item, dict):
        return item
    text = item.to_string() if hasattr(item, "to_string") else str(item)
    start = text.find("<component>")
    end = text.find("</component>", start)
    if start != -1 and end != -1:
        snippet = text[start + len("<component>") : end].strip()
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


class SuccessStructuredRunnable(Runnable):
    def invoke(self, input, config=None):
        return Result(
            attacks=[
                AgentAttack(
                    component="success",
                    component_uuid=uuid.uuid4(),
                )
            ]
        )

    async def ainvoke(self, input, config=None):
        return self.invoke(input, config=config)

    async def abatch(self, inputs, config=None, *args, **kwargs):
        return_exceptions = kwargs.get("return_exceptions", False)
        outputs = []
        for item in inputs:
            component_payload = _extract_component(item)
            outputs.append(
                Result(
                    attacks=[
                        AgentAttack(
                            component=component_payload.get("name", ""),
                            component_uuid=uuid.uuid4(),
                        )
                    ]
                )
            )
        return outputs


class FailingStructuredRunnable(Runnable):
    def invoke(self, input, config=None):
        raise RuntimeError("failure")

    async def ainvoke(self, input, config=None):
        raise RuntimeError("failure")

    async def abatch(self, inputs, config=None, *args, **kwargs):
        return_exceptions = kwargs.get("return_exceptions", False)
        if return_exceptions:
            return [RuntimeError("failure") for _ in inputs]
        raise RuntimeError("failure")


class DummyModel(BaseChatModel):
    def with_structured_output(self, schema, *args, **kwargs):
        return SuccessStructuredRunnable()

    def bind_tools(self, tools, **kwargs):
        return SuccessStructuredRunnable()

    def __ror__(self, other):
        # Support the 'prompt | model' chaining operator
        return self

    @property
    def _llm_type(self):
        # Return a dummy llm type
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Return a dummy response structure
        return {"generations": [], "llm_output": {}}


class FailingModel(BaseChatModel):
    def with_structured_output(self, schema, *args, **kwargs):
        return FailingStructuredRunnable()

    def bind_tools(self, tools, **kwargs):
        return FailingStructuredRunnable()

    def __ror__(self, other):
        return self

    @property
    def _llm_type(self):
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return {"generations": [], "llm_output": {}}


class MixedStructuredRunnable(Runnable):
    def __init__(self):
        self._next_uuid = uuid.uuid4()

    def invoke(self, input, config=None):
        raise NotImplementedError

    async def ainvoke(self, input, config=None):
        raise NotImplementedError

    async def abatch(self, inputs, config=None, *args, **kwargs):
        return_exceptions = kwargs.get("return_exceptions", False)
        outputs = []
        for item in inputs:
            component_payload = _extract_component(item)
            if component_payload.get("name") == "good":
                outputs.append(
                    Result(
                        attacks=[
                            AgentAttack(
                                component=component_payload.get("name", ""),
                                component_uuid=uuid.uuid4(),
                            )
                        ]
                    )
                )
            else:
                if return_exceptions:
                    outputs.append(RuntimeError("failure"))
                else:
                    raise RuntimeError("failure")
        return outputs


class MixedModel(BaseChatModel):
    def with_structured_output(self, schema, *args, **kwargs):
        return MixedStructuredRunnable()

    def bind_tools(self, tools, **kwargs):
        return MixedStructuredRunnable()

    def __ror__(self, other):
        return self

    @property
    def _llm_type(self):
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return {"generations": [], "llm_output": {}}


@pytest.fixture
def agent() -> MitreAttackAgent:
    return MitreAttackAgent(model=DummyModel())


def test_initialize_converts_uuids(monkeypatch, agent):
    state = AttackGraphStateModel(data_flow_report={"foo": "bar"})
    # Stub out convert_uuids_to_ids
    monkeypatch.setattr(
        agent.agent_helper, "convert_uuids_to_ids", lambda x: {"converted": True}
    )
    new_state = agent.initialize(state)
    assert new_state.data_flow_report == {"converted": True}


def test_finalize_converts_ids(monkeypatch, agent):
    state = AttackGraphStateModel()
    state.attacks = [{"a": 1}, {"b": 2}]
    # Stub out convert_ids_to_uuids
    monkeypatch.setattr(
        agent.agent_helper,
        "convert_ids_to_uuids",
        lambda threat: {"uuid_converted": threat},
    )
    new_state = agent.finalize(state)
    assert new_state.attacks == [
        {"uuid_converted": {"a": 1}},
        {"uuid_converted": {"b": 2}},
    ]


async def test_analyze_processes_components(agent):
    state = AttackGraphStateModel(
        data_flow_report={
            "processes": [
                {"name": "comp1"},
                {"name": "comp2"},
            ]
        }
    )

    result_state = await agent.analyze(state)

    assert len(result_state.attacks) == 2
    assert {attack.component for attack in result_state.attacks} == {
        "comp1",
        "comp2",
    }


async def test_analyze_handles_exceptions():
    failing_agent = MitreAttackAgent(model=FailingModel())
    state = AttackGraphStateModel(
        data_flow_report={"processes": [{"name": "comp"}]}
    )

    result_state = await failing_agent.analyze(state)
    assert result_state.attacks == []


async def test_analyze_with_none_report(agent):
    state = AttackGraphStateModel()
    state.data_flow_report = {}
    new_state = await agent.analyze(state)
    # When report is empty, attacks should be set to an empty list
    assert new_state.attacks == []


async def test_analyze_with_empty_report(agent):
    state = AttackGraphStateModel()
    # Default data_flow_report is an empty dict
    new_state = await agent.analyze(state)
    assert new_state.attacks == []


def test_get_workflow(agent):
    workflow = agent.get_workflow()
    # Ensure a workflow object is returned
    assert workflow is not None


async def test_analyze_integration_and_error():
    mixed_agent = MitreAttackAgent(model=MixedModel())
    state = AttackGraphStateModel(
        data_flow_report={
            "processes": [
                {"name": "good"},
                {"name": "bad"},
            ]
        }
    )

    new_state = await mixed_agent.analyze(state)
    assert len(new_state.attacks) == 1
    attack = new_state.attacks[0]
    assert attack.component == "good"
