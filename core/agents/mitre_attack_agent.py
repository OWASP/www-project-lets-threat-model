import logging
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from core.agents.agent_tools import AgentHelper
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from core.models.dtos.MitreAttack import AgentAttack
logger = logging.getLogger(__name__)

SYSTEM_MAP_PROMPT = """
You are a cybersecurity expert skilled in threat modeling and adversary behavior analysis using the MITRE ATT&CK framework.

You will be given a component from a system architecture, along with the complete system description in the form of a `DataFlowReport`. This report includes processes, data stores, external entities, data flows, and trust boundaries that define the architecture and interactions within the system.

Your task is to:
1. Analyze the provided **component** in the context of the full system.
2. Identify the relevant MITRE ATT&CK **tactics** and **techniques** that an adversary could use to compromise this component, move laterally, or exfiltrate data.
3. For each identified technique, include:
   - The **component** name this technique targets,
   - The **component_uuid** (UUID) for that component,
   - The **attack_tactic** (e.g., Initial Access, Execution),
   - The **technique_id** and **technique_name** (e.g., T1078 · Valid Accounts),
   - The **url** of the ATT&CK page for the technique or sub‑technique you are citing,
   - A concise **reason_for_relevance** explaining why the technique applies,
   - List one or more concrete **mitigation** recommendations,
   - The flag **is_subtechnique** (`true` or `false`);
     if `true`, also provide **parent_id** and **parent_name**.

If a technique is a *sub‑technique*, mention the parent technique first and indent the sub‑technique beneath it (e.g., “T1059: Command & Scripting Interpreter” → “  • T1059.003 Windows Command Shell”).

Focus solely on the current component provided in context, but leverage insights from the full system report as needed and use `https://attack.mitre.org/techniques/enterprise/` get the latest information on tactics and techniques.
Do not provide generic advice or high-level summaries; focus on specific techniques that could be applied to the component based on the system architecture.
"""


class AttackGraphStateModel(BaseModel):
    data_flow_report: Dict[str, Any] = Field(default_factory=dict)
    attacks: List[AgentAttack] = Field(default_factory=list)


class Result(BaseModel):
    """LLM tool schema describing MITRE ATT&CK techniques discovered for a component."""

    attacks: List[AgentAttack] = Field(
        default_factory=list,
        description="List of identified attacks based on the MITRE ATT&CK framework.",
    )


class MitreAttackAgent:
    """Agent to analyze system components and identify MITRE ATT&CK techniques."""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.agent_helper = AgentHelper()

    def initialize(self, state: AttackGraphStateModel) -> AttackGraphStateModel:
        """Initialize the state by converting UUIDs to internal numbered IDs."""
        logger.info("Initializing threat modeling state...")

        state.data_flow_report = self.agent_helper.convert_uuids_to_ids(
            state.data_flow_report
        )

        return state

    def finalize(self, state: AttackGraphStateModel) -> AttackGraphStateModel:
        """Clean up the state by converting internal numbered IDs back to UUIDs."""
        logger.info("Cleaning up threat modeling state...")

        state.attacks = [
            self.agent_helper.convert_ids_to_uuids(attack) for attack in state.attacks
        ]

        return state

    def _log_attack_summary(self, attacks: List[AgentAttack]) -> None:
        """Log summary of identified MITRE ATT&CK techniques by tactic."""
        if not attacks:
            logger.warning("No attacks identified.")
            return

        tactic_counts = {}
        technique_counts = {}

        for attack in attacks:
            tactic = getattr(attack, "attack_tactic", "Unknown")
            technique = f"{attack.technique_id}: {attack.technique_name}"

            tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
            technique_counts[technique] = technique_counts.get(technique, 0) + 1

        logger.info(
            f"🛡️ MITRE ATT&CK Summary:\n"
            f"  📊 Total Techniques Identified: {len(attacks)}\n"
            f"  🎯 Tactics Breakdown:"
        )
        for tactic, count in sorted(tactic_counts.items(), key=lambda x: x[0]):
            logger.info(f"    - {tactic}: {count}")

        logger.debug("  🧰 Techniques Identified:")
        for technique, count in sorted(
            technique_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.debug(f"    - {technique}: {count}")

    async def analyze(self, state: AttackGraphStateModel) -> AttackGraphStateModel:
        report = state.data_flow_report
        if not report:  # Ensure report is not empty
            logger.error("DataFlowReport is empty in generate_attack_map.")
            state.attacks = []
            return state

        system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MAP_PROMPT)
        user_prompt = HumanMessagePromptTemplate.from_template(
            """
            <data_flow_report>
            {data_flow_report}
            </data_flow_report>

            <component>
            {component}
            </component>
            """
        )
        prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        logger.debug(
            "🗂️ Preparing analysis for %d components",
            sum(
                len(report.get(k, []))
                for k in (
                    "external_entities",
                    "processes",
                    "data_stores",
                    "trust_boundaries",
                    "data_flows",
                )
            ),
        )
        structured_model = self.model.with_structured_output(
            Result, method="function_calling"
        )
        chain = prompt | structured_model

        components: list[Dict[str, Any]] = []
        for key in (
            "external_entities",
            "processes",
            "data_stores",
            "trust_boundaries",
            "data_flows",
        ):
            components.extend(sorted(report.get(key, []), key=lambda c: c.get("name", "")))

        if not components:
            logger.warning("No components found for MITRE ATT&CK analysis.")
            state.attacks = []
            return state

        inputs = [
            {
                "data_flow_report": json.dumps(report, sort_keys=True),
                "component": json.dumps(component, sort_keys=True),
            }
            for component in components
        ]

        results = await chain.abatch(inputs, return_exceptions=True)

        attacks: List[AgentAttack] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error("❌ An error occurred in component analysis", exc_info=res)
                continue
            try:
                parsed = res if isinstance(res, Result) else Result.model_validate(res)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to parse MITRE analysis result: %s", exc)
                continue
            attacks.extend(parsed.attacks)

        logger.info(
            "✅ Finished MITRE ATT&CK analysis. Total attacks found: %d",
            len(attacks),
        )
        self._log_attack_summary(attacks)

        state.attacks = attacks
        return state

    def get_workflow(self):
        workflow = StateGraph(AttackGraphStateModel)

        workflow.add_node("initialize", self.initialize)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("finalize", self.finalize)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "analyze")
        workflow.add_edge("analyze", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()
