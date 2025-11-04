import json

from langchain_core.runnables import Runnable

from core.agents.threat_model_agent import ThreatModelAgent, ThreatGraphStateModel
from langchain_core.language_models.chat_models import BaseChatModel
from core.models.dtos.Threat import AgentThreat


def _extract_component_name(item) -> str:
    if isinstance(item, dict):
        return item.get("component", "").get("name", "") if isinstance(item.get("component"), dict) else item.get("component", "")
    text = ""
    if hasattr(item, "to_string"):
        text = item.to_string()
    else:
        text = str(item)
    start = text.find("<component>")
    end = text.find("</component>", start)
    if start != -1 and end != -1:
        snippet = text[start + len("<component>") : end].strip()
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data.get("name", "")
        except json.JSONDecodeError:
            pass
        return snippet
    return ""


class DummyRunnable(Runnable):
    def invoke(self, input, config=None):
        return {"threats": [{"name": "dummy"}]}

    async def ainvoke(self, input, config=None):
        return {"threats": [{"name": "dummy"}]}

    async def abatch(self, inputs, config=None, *args, **kwargs):
        return_exceptions = kwargs.get("return_exceptions", False)
        outputs = []
        for item in inputs:
            component_name = _extract_component_name(item)
            if not component_name:
                if return_exceptions:
                    outputs.append(RuntimeError("missing component"))
                    continue
                raise RuntimeError("missing component")
            outputs.append(
                {
                    "threats": [
                        AgentThreat(
                            name=component_name,
                            component_names=[component_name],
                        )
                    ]
                }
            )
        return outputs


class DummyModel(BaseChatModel):
    def with_structured_output(self, schema, *args, **kwargs):
        return DummyRunnable()

    def bind_tools(self, tools, **kwargs):
        return DummyRunnable()

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


def test_initialize(monkeypatch):
    agent = ThreatModelAgent(model=DummyModel())
    state = ThreatGraphStateModel(asset={"foo": "bar"}, data_flow_report={"baz": 1})
    # Stub out convert_uuids_to_ids to verify it's called correctly
    monkeypatch.setattr(
        agent.agent_helper, "convert_uuids_to_ids", lambda x: f"converted-{repr(x)}"
    )

    new_state = agent.initialize(state)

    assert new_state.asset == "converted-{'foo': 'bar'}"
    assert new_state.data_flow_report == "converted-{'baz': 1}"


def test_finalize(monkeypatch):
    agent = ThreatModelAgent(model=DummyModel())
    state = ThreatGraphStateModel()
    # Pretend we already have two raw threat dicts
    state.threats = [{"a": 1}, {"b": 2}]
    # Stub out convert_ids_to_uuids
    monkeypatch.setattr(
        agent.agent_helper,
        "convert_ids_to_uuids",
        lambda threat: f"uuid-{repr(threat)}",
    )

    new_state = agent.finalize(state)

    assert new_state.threats == [
        "uuid-{'a': 1}",
        "uuid-{'b': 2}",
    ]


async def test_analyze_aggregates_all_components():
    agent = ThreatModelAgent(model=DummyModel())
    # Create a state with two external_entities and no other components
    state = ThreatGraphStateModel(
        asset={"unused": True},
        data_flow_report={
            "external_entities": [
                {"name": "CompA"},
                {"name": "CompB"},
            ],
            "processes": [],
            "data_stores": [],
            "trust_boundaries": [],
        },
    )
    # Run the async analyze method
    result_state = await agent.analyze(state)

    # Verify threats were aggregated for both components
    names = {threat.name for threat in result_state.threats}
    assert names == {"CompA", "CompB"}
