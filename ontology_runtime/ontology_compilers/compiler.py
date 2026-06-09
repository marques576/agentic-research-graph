"""
Abstract compiler interface and compilation result for the Ontology Runtime.

Every compiler transforms an Ontology into a specific target format (e.g.
NetworkX DiGraph, OWL/XML, RDF/XML, CYPHER queries).
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ..ontology_model.types import Ontology


@dataclass
class CompilationResult:
    """Output from a compiler.

    Attributes
    ----------
    output : Any
        The compiled artefact (varies by compiler).
    format_name : str
        Identifier for the output format, e.g. "networkx", "owl", "rdf".
    compilation_time_ms : float
        Wall-clock time for the compilation in milliseconds.
    metadata : dict
        Extra info such as node/edge counts, warnings, etc.
    """

    output: Any = None
    format_name: str = ""
    compilation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Compiler(abc.ABC):
    """Abstract base for all ontology compilers.

    Subclasses must implement :meth:`compile`.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abc.abstractmethod
    def compile(self, ontology: Ontology) -> CompilationResult:
        """Compile *ontology* into a target format.

        Parameters
        ----------
        ontology : Ontology

        Returns
        -------
        CompilationResult
        """
        ...
