"""Token budget guard for agent calls (CLAUDE.md, "Бюджет токенов"; docs/BUDGET.md)."""

from qlab.budget.agent import AgentResult, probe, run_agent
from qlab.budget.guard import (
    BudgetExhausted,
    CandidateBudget,
    CandidateExhausted,
    ExplicitNoBudget,
    NightBudget,
    NightExhausted,
    Stage,
    calibrate,
)

__all__ = [
    "AgentResult",
    "BudgetExhausted",
    "CandidateBudget",
    "CandidateExhausted",
    "ExplicitNoBudget",
    "NightBudget",
    "NightExhausted",
    "Stage",
    "calibrate",
    "probe",
    "run_agent",
]
