"""Shared role-label utilities.

The training loop and analysis pipeline must use the exact same
definition of role to be directly comparable. Keep this module as
the single source of truth.

Priority (matches KarmaAgent._infer_role):
    BEING_ZAPPED (victim)   > ZAP_AGENT (aggressor)
                            > ZAP_WASTE (cleaner)
                            > APPLE_EATEN (forager)
                            > NEUTRAL
"""

from __future__ import annotations

from typing import Dict, List, Tuple


ROLE_NEUTRAL = 0
ROLE_AGGRESSOR = 1
ROLE_VICTIM = 2
ROLE_CLEANER = 3
ROLE_FORAGER = 4

ROLE_NAMES = {
    ROLE_NEUTRAL: "NEUTRAL",
    ROLE_AGGRESSOR: "ZAP_AGENT",
    ROLE_VICTIM: "BEING_ZAPPED",
    ROLE_CLEANER: "ZAP_WASTE",
    ROLE_FORAGER: "APPLE_EATEN",
}

NAME_TO_ROLE = {v: k for k, v in ROLE_NAMES.items()}


def infer_role(events: List[Dict], agent_id: str) -> int:
    """Return single-label role index using the priority rule.

    Returns 0 (NEUTRAL) when no relevant event fires this step.
    """
    is_victim = False
    is_aggressor = False
    is_cleaner = False
    is_forager = False

    for e in events:
        etype = e.get("type", "")
        if etype == "BEING_ZAPPED" and e.get("victim") == agent_id:
            is_victim = True
        elif etype == "ZAP_AGENT" and e.get("attacker") == agent_id:
            is_aggressor = True
        elif etype == "ZAP_WASTE" and e.get("actor") == agent_id:
            is_cleaner = True
        elif etype == "APPLE_EATEN" and e.get("actor") == agent_id:
            is_forager = True

    if is_victim:
        return ROLE_VICTIM
    if is_aggressor:
        return ROLE_AGGRESSOR
    if is_cleaner:
        return ROLE_CLEANER
    if is_forager:
        return ROLE_FORAGER
    return ROLE_NEUTRAL


def infer_role_multilabel(events: List[Dict], agent_id: str) -> Tuple[int, int, int, int, int]:
    """Return a 5-tuple of 0/1 flags for (NEUTRAL, AGGRESSOR, VICTIM, CLEANER, FORAGER).

    NEUTRAL is set iff no other role fires.
    """
    agg = vic = cln = fgr = 0
    for e in events:
        etype = e.get("type", "")
        if etype == "BEING_ZAPPED" and e.get("victim") == agent_id:
            vic = 1
        elif etype == "ZAP_AGENT" and e.get("attacker") == agent_id:
            agg = 1
        elif etype == "ZAP_WASTE" and e.get("actor") == agent_id:
            cln = 1
        elif etype == "APPLE_EATEN" and e.get("actor") == agent_id:
            fgr = 1
    neutral = 1 if (agg + vic + cln + fgr) == 0 else 0
    return (neutral, agg, vic, cln, fgr)
