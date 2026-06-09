"""Proposal-driven ontology changes.

All mutations to the ontology schema go through proposals.  A proposal
collects a set of operations that are validated before being applied to
the active workspace ontology.

Uses ``OntologyOperation`` from the ``ontology_operations`` package for
the actual operation definitions and application.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ontology_runtime.ontology_operations import (
    OntologyOperation,
    OntologyTransaction,
    apply_operation,
)


# ---------------------------------------------------------------------------
# ProposalStatus
# ---------------------------------------------------------------------------


class ProposalStatus(Enum):
    """Lifecycle states an ontology proposal can be in."""

    DRAFT = "draft"
    PENDING = "pending"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


# ---------------------------------------------------------------------------
# OntologyProposal
# ---------------------------------------------------------------------------


@dataclass
class OntologyProposal:
    """A proposed set of changes to the ontology schema.

    Attributes
    ----------
    id : str
        Unique proposal identifier.
    title : str
        Short human-readable summary.
    description : str
        Longer explanation of the proposed changes.
    changes : list[OntologyOperation]
        The individual operations that make up this proposal.
    status : ProposalStatus
        Current lifecycle state.
    author : str
        Who / what created the proposal.
    created_at : float
        Unix timestamp of creation.
    reviewed_at : float | None
        Unix timestamp of when the proposal was reviewed (approved/rejected).
    review_notes : str
        Optional notes left by the reviewer.
    """

    id: str = ""
    title: str = ""
    description: str = ""
    changes: list[OntologyOperation] = field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    author: str = "system"
    created_at: float = field(default_factory=time.time)
    reviewed_at: float | None = None
    review_notes: str = ""


# ---------------------------------------------------------------------------
# ProposalManager
# ---------------------------------------------------------------------------


class ProposalManager:
    """Manages the lifecycle of ontology proposals.

    Delegates validation to an optional *validation_engine* callable and
    applies approved proposals via ``OntologyTransaction`` from the
    ``ontology_operations`` package.

    Parameters
    ----------
    validation_engine : Callable | None
        An optional callable that receives a proposal and returns
        ``(is_valid: bool, notes: str)``.  If ``None`` validation
        always passes.
    """

    def __init__(
        self,
        validation_engine: Callable[[OntologyProposal], tuple[bool, str]] | None = None,
    ) -> None:
        self._proposals: dict[str, OntologyProposal] = {}
        self._validation_engine = validation_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_proposal(
        self,
        title: str,
        changes: list[OntologyOperation],
        author: str = "system",
        description: str = "",
    ) -> OntologyProposal:
        """Create a new proposal in DRAFT status.

        Parameters
        ----------
        title : str
            Short human-readable title.
        changes : list[OntologyOperation]
            The operations that this proposal bundles.
        author : str
            Creator label.
        description : str
            Optional longer description.

        Returns
        -------
        OntologyProposal
            The newly created proposal.
        """
        proposal = OntologyProposal(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            changes=list(changes),
            status=ProposalStatus.DRAFT,
            author=author,
            created_at=time.time(),
        )
        self._proposals[proposal.id] = proposal
        return proposal

    def validate_proposal(self, proposal_id: str) -> tuple[bool, str]:
        """Run validation on a proposal.

        Transitions the proposal to ``VALIDATED`` if validation passes,
        otherwise it stays at its current status.

        Parameters
        ----------
        proposal_id : str
            Identifier of the proposal to validate.

        Returns
        -------
        (is_valid: bool, notes: str)

        Raises
        ------
        ValueError
            If the proposal does not exist or is not in a validatable
            status (must be DRAFT or PENDING).
        """
        proposal = self._get_proposal(proposal_id)
        if proposal.status not in (ProposalStatus.DRAFT, ProposalStatus.PENDING):
            raise ValueError(
                f"Cannot validate proposal {proposal_id!r} in status "
                f"{proposal.status.value!r}.  Must be DRAFT or PENDING."
            )

        if self._validation_engine:
            is_valid, notes = self._validation_engine(proposal)
        else:
            is_valid, notes = True, "No validation engine configured — skipped."

        if is_valid:
            proposal.status = ProposalStatus.VALIDATED
        proposal.review_notes = notes
        proposal.reviewed_at = time.time()
        return is_valid, notes

    def approve_proposal(self, proposal_id: str, notes: str = "") -> OntologyProposal:
        """Approve a validated proposal, transitioning it to APPROVED.

        Parameters
        ----------
        proposal_id : str
            Identifier of the proposal.
        notes : str
            Optional reviewer notes.

        Returns
        -------
        OntologyProposal
            The updated proposal.

        Raises
        ------
        ValueError
            If the proposal does not exist or is not in VALIDATED status.
        """
        proposal = self._get_proposal(proposal_id)
        if proposal.status != ProposalStatus.VALIDATED:
            raise ValueError(
                f"Cannot approve proposal {proposal_id!r} in status "
                f"{proposal.status.value!r}.  Must be VALIDATED."
            )
        proposal.status = ProposalStatus.APPROVED
        proposal.reviewed_at = time.time()
        if notes:
            proposal.review_notes = notes
        return proposal

    def reject_proposal(
        self,
        proposal_id: str,
        reason: str = "",
    ) -> OntologyProposal:
        """Reject a proposal, transitioning it to REJECTED.

        Parameters
        ----------
        proposal_id : str
            Identifier of the proposal.
        reason : str
            Reason for rejection.

        Returns
        -------
        OntologyProposal
            The updated proposal.

        Raises
        ------
        ValueError
            If the proposal does not exist.
        """
        proposal = self._get_proposal(proposal_id)
        proposal.status = ProposalStatus.REJECTED
        proposal.reviewed_at = time.time()
        proposal.review_notes = reason
        return proposal

    def apply_proposal(
        self,
        proposal_id: str,
        ontology: Any,
    ) -> list[OntologyOperation]:
        """Apply an approved proposal's operations to an ontology.

        Uses ``OntologyTransaction`` to commit all operations atomically.

        Parameters
        ----------
        proposal_id : str
            Identifier of the approved proposal.
        ontology : Any
            The ontology instance to mutate in-place.

        Returns
        -------
        list[OntologyOperation]
            The list of operations that were applied.

        Raises
        ------
        ValueError
            If the proposal does not exist or is not in APPROVED status.
        """
        proposal = self._get_proposal(proposal_id)
        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError(
                f"Cannot apply proposal {proposal_id!r} in status "
                f"{proposal.status.value!r}.  Must be APPROVED."
            )

        # Use OntologyTransaction to apply all operations atomically
        tx = OntologyTransaction()
        for op in proposal.changes:
            tx.add_operation(op)
        applied = tx.commit(ontology)

        proposal.status = ProposalStatus.APPLIED
        return applied

    def list_proposals(
        self,
        status_filter: ProposalStatus | None = None,
    ) -> list[OntologyProposal]:
        """List proposals, optionally filtered by status.

        Parameters
        ----------
        status_filter : ProposalStatus | None
            If set, only proposals with this status are returned.

        Returns
        -------
        list[OntologyProposal]
            Matching proposals (newest first).
        """
        proposals = list(self._proposals.values())
        if status_filter:
            proposals = [p for p in proposals if p.status == status_filter]
        proposals.sort(key=lambda p: p.created_at, reverse=True)
        return proposals

    def get_proposal(self, proposal_id: str) -> OntologyProposal | None:
        """Retrieve a single proposal by identifier.

        Parameters
        ----------
        proposal_id : str
            Proposal UUID.

        Returns
        -------
        OntologyProposal | None
        """
        return self._proposals.get(proposal_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_proposal(self, proposal_id: str) -> OntologyProposal:
        if proposal_id not in self._proposals:
            raise ValueError(f"Proposal {proposal_id!r} does not exist.")
        return self._proposals[proposal_id]
