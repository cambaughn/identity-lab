"""Temporal identity stabilization — per-track voting and hysteresis.

Raw per-frame matches flicker; this layer turns them into a calm decision:

- promotion: a track shows UNKNOWN until PROMOTE_N consecutive agreeing
  known-matches arrive, then the name is displayed (CONFIRMED)
- hysteresis: once a name is displayed, a *different* name must win
  SWITCH_N consecutive observations before the label switches
- decay: DEMOTE_N consecutive non-matching observations drop the label
  back to UNKNOWN
- reset: when a track dies (person left), its state is forgotten entirely

Pure logic, no Qt. Emits the spec IdentityObservation per observation.
"""

from dataclasses import dataclass
from datetime import datetime

from identity_lab.identity.matcher import MatchResult
from identity_lab.identity.types import IdentityObservation

PROMOTE_N = 3   # consecutive agreeing matches to announce a name
SWITCH_N = 5    # consecutive matches for a *different* name to displace one
DEMOTE_N = 6    # consecutive non-matches to fall back to UNKNOWN


@dataclass
class _TrackState:
    current_id: str | None = None
    current_name: str | None = None
    candidate_id: str | None = None
    candidate_name: str | None = None
    candidate_count: int = 0
    miss_count: int = 0


class IdentityStabilizer:
    def __init__(
        self,
        promote_n: int = PROMOTE_N,
        switch_n: int = SWITCH_N,
        demote_n: int = DEMOTE_N,
    ) -> None:
        self._promote_n = promote_n
        self._switch_n = switch_n
        self._demote_n = demote_n
        self._states: dict[str, _TrackState] = {}

    def observe(
        self,
        track_id: str,
        match: MatchResult | None,
        bbox: tuple[int, int, int, int],
    ) -> IdentityObservation:
        state = self._states.setdefault(track_id, _TrackState())

        matched_id = match.identity_id if match is not None and match.is_known else None
        matched_name = match.display_name if match is not None and match.is_known else None

        if matched_id is not None:
            reason = self._observe_known(state, matched_id, matched_name)
        else:
            reason = self._observe_unknown(
                state, match.reason if match is not None else "NO MATCH DATA"
            )

        similarity = match.similarity if match is not None else None
        second = (
            match.second_best.score
            if match is not None and match.second_best is not None
            else None
        )
        return IdentityObservation(
            track_id=track_id,
            identity_id=state.current_id,
            display_name=state.current_name,
            similarity=similarity,
            second_best_similarity=second,
            is_known=state.current_id is not None,
            reason=reason,
            bbox=bbox,
            timestamp=datetime.now(),
        )

    def _observe_known(self, state: _TrackState, ident: str, name: str) -> str:
        state.miss_count = 0
        if state.current_id == ident:
            state.candidate_id = None
            state.candidate_count = 0
            return "CONFIRMED"
        # A (different) known identity observed.
        if state.candidate_id == ident:
            state.candidate_count += 1
        else:
            state.candidate_id = ident
            state.candidate_name = name
            state.candidate_count = 1
        if state.current_id is None:
            if state.candidate_count >= self._promote_n:
                state.current_id = ident
                state.current_name = name
                state.candidate_id = None
                state.candidate_count = 0
                return "CONFIRMED"
            return f"PENDING {name.upper()} {state.candidate_count}/{self._promote_n}"
        # Hysteresis: displayed name only yields after SWITCH_N straight wins.
        if state.candidate_count >= self._switch_n:
            state.current_id = ident
            state.current_name = name
            state.candidate_id = None
            state.candidate_count = 0
            return "SWITCHED"
        return (
            f"HOLD — CANDIDATE {name.upper()} "
            f"{state.candidate_count}/{self._switch_n}"
        )

    def _observe_unknown(self, state: _TrackState, raw_reason: str) -> str:
        state.candidate_id = None
        state.candidate_count = 0
        if state.current_id is None:
            return raw_reason
        state.miss_count += 1
        if state.miss_count >= self._demote_n:
            state.current_id = None
            state.current_name = None
            state.miss_count = 0
            return "DECAYED TO UNKNOWN"
        return f"HOLD {state.miss_count}/{self._demote_n}"

    def is_confirmed(self, track_id: str) -> bool:
        """True when this track currently displays a name. Unseen tracks are
        unconfirmed — used to run recognition eagerly until names lock in."""
        state = self._states.get(track_id)
        return state is not None and state.current_id is not None

    def forget(self, track_id: str) -> None:
        """Track died — the person left; a return starts from scratch."""
        self._states.pop(track_id, None)

    def reset(self) -> None:
        """Drop all state (camera stop or gallery change)."""
        self._states.clear()

    @property
    def tracked_count(self) -> int:
        return len(self._states)
