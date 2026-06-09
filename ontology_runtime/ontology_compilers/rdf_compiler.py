"""
RDF compiler — produces RDF/XML (Resource Description Framework) representation.

Generates a minimal RDF graph where each ontology object is a resource
typed by its id, and each relation is an RDF statement.
"""

from __future__ import annotations

import time
from xml.sax.saxutils import escape as xml_escape

from .compiler import CompilationResult, Compiler
from ..ontology_model.types import Ontology


class RDFCompiler(Compiler):
    """Compiles an Ontology into an RDF/XML string.

    Parameters
    ----------
    base_uri : str
        Base URI for resources (default: "http://example.org/ontology").
    name : str
        Optional label.
    """

    def __init__(
        self,
        base_uri: str = "http://example.org/ontology",
        name: str = "",
    ) -> None:
        super().__init__(name=name or "RDFCompiler")
        self._base_uri = base_uri.rstrip("/")

    def compile(self, ontology: Ontology) -> CompilationResult:
        """Generate RDF/XML from *ontology*.

        Parameters
        ----------
        ontology : Ontology

        Returns
        -------
        CompilationResult
            ``.output`` is a string containing valid RDF/XML.
        """
        start = time.perf_counter()

        lines: list[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            '<rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
            f'xmlns:onto="{self._base_uri}#">'
        )
        lines.append("")

        for obj_id, obj in ontology.objects.items():
            oid_safe = xml_escape(obj_id.replace(" ", "_"))
            lines.append(
                f'  <rdf:Description rdf:about="{self._base_uri}#{oid_safe}">'
            )
            lines.append(
                f'    <rdf:type rdf:resource="{self._base_uri}#{oid_safe}"/>'
            )
            lines.append(
                f'    <rdfs:label>{xml_escape(obj.name or obj_id)}</rdfs:label>'
            )
            if obj.parent_id:
                parent = xml_escape(obj.parent_id.replace(" ", "_"))
                lines.append(
                    f'    <rdfs:subClassOf rdf:resource="{self._base_uri}#{parent}"/>'
                )
            for prop in obj.properties:
                pk = xml_escape(prop.name.replace(" ", "_"))
                pv = xml_escape(str(prop.default) if prop.default is not None else "")
                lines.append(f'    <onto:{pk}>{pv}</onto:{pk}>')
            lines.append("  </rdf:Description>")
            lines.append("")

        for rel_id, rel in ontology.relations.items():
            src = xml_escape(rel.source_type.replace(" ", "_"))
            tgt = xml_escape(rel.target_type.replace(" ", "_"))
            rname = xml_escape(rel.name.replace(" ", "_") if rel.name else rel_id.replace(" ", "_"))
            lines.append(
                f'  <rdf:Description rdf:about="{self._base_uri}#{src}">'
            )
            lines.append(
                f'    <onto:{rname} rdf:resource="{self._base_uri}#{tgt}"/>'
            )
            lines.append("  </rdf:Description>")
            lines.append("")

        lines.append("</rdf:RDF>")

        output = "\n".join(lines)
        elapsed = (time.perf_counter() - start) * 1000

        return CompilationResult(
            output=output,
            format_name="rdf",
            compilation_time_ms=elapsed,
            metadata={
                "object_count": len(ontology.objects),
                "relation_count": len(ontology.relations),
            },
        )
