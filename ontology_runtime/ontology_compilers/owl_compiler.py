"""
OWL compiler — produces OWL/XML (Web Ontology Language) representation.

Output is a minimal OWL 2 DL ontology in RDF/XML syntax describing the
class hierarchy and object properties.
"""

from __future__ import annotations

import time
from xml.sax.saxutils import escape as xml_escape

from .compiler import CompilationResult, Compiler
from ..ontology_model.types import Ontology


class OWLCompiler(Compiler):
    """Compiles an Ontology into an OWL/XML string.

    Parameters
    ----------
    ontology_iri : str
        Base IRI for the ontology (default: "http://example.org/ontology").
    name : str
        Optional label.
    """

    def __init__(
        self,
        ontology_iri: str = "http://example.org/ontology",
        name: str = "",
    ) -> None:
        super().__init__(name=name or "OWLCompiler")
        self._ontology_iri = ontology_iri.rstrip("/")

    def compile(self, ontology: Ontology) -> CompilationResult:
        """Generate OWL/XML from *ontology*.

        Parameters
        ----------
        ontology : Ontology

        Returns
        -------
        CompilationResult
            ``.output`` is a string containing valid OWL/XML.
        """
        start = time.perf_counter()

        lines: list[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            f'<rdf:RDF '
            f'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            f'xmlns:owl="http://www.w3.org/2002/07/owl#" '
            f'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
            f'xmlns:xsd="http://www.w3.org/2001/XMLSchema#" '
            f'xmlns:onto="{self._ontology_iri}#">'
        )
        lines.append("")

        lines.append(f'  <owl:Ontology rdf:about="{self._ontology_iri}"/>')
        lines.append("")

        # Object types as OWL classes
        for obj_id, obj in ontology.objects.items():
            oid_safe = xml_escape(obj_id.replace(" ", "_"))
            lines.append(
                f'  <owl:Class rdf:about="{self._ontology_iri}#{oid_safe}"/>'
            )

        # Hierarchy (parent_id -> subclass)
        for obj in ontology.objects.values():
            if obj.parent_id:
                child = xml_escape(obj.id.replace(" ", "_"))
                parent = xml_escape(obj.parent_id.replace(" ", "_"))
                lines.append(
                    f'  <owl:Class rdf:about="{self._ontology_iri}#{child}">'
                )
                lines.append(
                    f'    <rdfs:subClassOf rdf:resource="{self._ontology_iri}#{parent}"/>'
                )
                lines.append("  </owl:Class>")

        # Relation types as OWL ObjectProperties
        for rel_id, rel in ontology.relations.items():
            rname = xml_escape(rel.name.replace(" ", "_") if rel.name else rel_id.replace(" ", "_"))
            lines.append(
                f'  <owl:ObjectProperty '
                f'rdf:about="{self._ontology_iri}#{rname}"/>'
            )

        lines.append("")
        lines.append("</rdf:RDF>")

        output = "\n".join(lines)
        elapsed = (time.perf_counter() - start) * 1000

        return CompilationResult(
            output=output,
            format_name="owl",
            compilation_time_ms=elapsed,
            metadata={
                "object_count": len(ontology.objects),
                "relation_count": len(ontology.relations),
            },
        )
