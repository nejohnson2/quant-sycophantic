"""Formal causal framework: DAG specification, identification, and TikZ rendering."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    """A node in the causal DAG."""

    name: str
    label: str
    observed: bool = True
    description: str = ""


@dataclass
class CausalDAG:
    """Directed acyclic graph encoding assumed causal structure.

    Provides backdoor criterion identification, adjustment set computation,
    and TikZ rendering for the paper.
    """

    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add_node(
        self,
        name: str,
        label: str,
        observed: bool = True,
        description: str = "",
    ) -> None:
        self.nodes[name] = CausalNode(name, label, observed, description)

    def add_edge(self, source: str, target: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source} -> {target}")
        self.edges.append((source, target))

    def parents(self, node: str) -> set[str]:
        return {src for src, tgt in self.edges if tgt == node}

    def children(self, node: str) -> set[str]:
        return {tgt for src, tgt in self.edges if src == node}

    def descendants(self, node: str) -> set[str]:
        visited = set()
        queue = list(self.children(node))
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(self.children(current))
        return visited

    def ancestors(self, node: str) -> set[str]:
        visited = set()
        queue = list(self.parents(node))
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(self.parents(current))
        return visited

    def _all_directed_paths(
        self, source: str, target: str
    ) -> list[list[str]]:
        """Find all directed paths from source to target."""
        paths = []
        stack = [(source, [source])]
        while stack:
            current, path = stack.pop()
            if current == target and len(path) > 1:
                paths.append(path)
                continue
            for child in self.children(current):
                if child not in path:  # avoid cycles
                    stack.append((child, path + [child]))
        return paths

    def backdoor_adjustment_set(
        self,
        treatment: str,
        outcome: str,
    ) -> dict:
        """Compute a valid adjustment set using the backdoor criterion.

        The backdoor criterion requires a set Z such that:
        1. No node in Z is a descendant of treatment
        2. Z blocks every backdoor path (non-causal path) from treatment to outcome

        Returns dict with adjustment set and reasoning.
        """
        # Identify backdoor paths: paths into treatment (not through its children)
        treatment_parents = self.parents(treatment)
        treatment_descendants = self.descendants(treatment)

        # Candidate adjustment variables: observed non-descendants of treatment
        candidates = {
            name
            for name, node in self.nodes.items()
            if node.observed
            and name != treatment
            and name != outcome
            and name not in treatment_descendants
        }

        # Confounders: nodes that are ancestors of both treatment and outcome
        # (or that open backdoor paths)
        outcome_ancestors = self.ancestors(outcome)
        treatment_ancestors = self.ancestors(treatment)
        confounders = candidates & (outcome_ancestors | treatment_ancestors)

        # The sufficient adjustment set is all observed confounders
        adjustment_set = confounders

        # Also identify mediators (on causal paths from treatment to outcome)
        causal_paths = self._all_directed_paths(treatment, outcome)
        mediators = set()
        for path in causal_paths:
            for node in path[1:-1]:  # exclude treatment and outcome
                mediators.add(node)

        return {
            "adjustment_set": adjustment_set,
            "mediators": mediators,
            "unobserved_confounders": {
                name
                for name, node in self.nodes.items()
                if not node.observed
                and name in (outcome_ancestors & treatment_ancestors)
            },
            "causal_paths": causal_paths,
            "treatment_descendants": treatment_descendants,
            "reasoning": _format_identification_reasoning(
                treatment, outcome, adjustment_set, mediators,
                {n for n, nd in self.nodes.items() if not nd.observed},
            ),
        }

    def report_identification(
        self,
        treatment: str,
        outcome: str,
    ) -> str:
        """Generate a human-readable identification report."""
        result = self.backdoor_adjustment_set(treatment, outcome)

        lines = [
            "=" * 60,
            "CAUSAL IDENTIFICATION REPORT",
            "=" * 60,
            f"Treatment: {treatment} ({self.nodes[treatment].label})",
            f"Outcome:   {outcome} ({self.nodes[outcome].label})",
            "",
            "--- Assumed Causal Structure ---",
        ]
        for src, tgt in self.edges:
            obs_src = "" if self.nodes[src].observed else " [unobserved]"
            obs_tgt = "" if self.nodes[tgt].observed else " [unobserved]"
            lines.append(
                f"  {self.nodes[src].label}{obs_src} -> "
                f"{self.nodes[tgt].label}{obs_tgt}"
            )

        lines += [
            "",
            "--- Identification ---",
            f"Backdoor adjustment set: {{{', '.join(sorted(result['adjustment_set']))}}}",
            f"Mediators (on causal paths): {{{', '.join(sorted(result['mediators']))}}}",
            f"Unobserved confounders: {{{', '.join(sorted(result['unobserved_confounders']))}}}",
            "",
            "--- Estimands ---",
            "Total effect:              Do NOT condition on mediators",
            f"  Adjust for: {{{', '.join(sorted(result['adjustment_set']))}}}",
            "Controlled direct effect:  Condition on mediators",
            f"  Adjust for: {{{', '.join(sorted(result['adjustment_set'] | result['mediators']))}}}",
            "",
            "--- Identifying Assumptions ---",
            "1. SUTVA: No interference between battles (battles are independent)",
            "2. Positivity: P(S=s | X=x) > 0 for all x in support",
            "   (Checked via propensity score overlap diagnostics)",
            "3. Conditional ignorability: Y(s) ⊥ S | X",
            f"   where X = {{{', '.join(sorted(result['adjustment_set']))}}}",
        ]
        if result["unobserved_confounders"]:
            lines += [
                "",
                "⚠ WARNING: Unobserved confounders present:",
                f"  {{{', '.join(sorted(result['unobserved_confounders']))}}}",
                "  Conditional ignorability CANNOT be verified.",
                "  Use E-values and Rosenbaum bounds to assess sensitivity.",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_tikz(
        self,
        treatment: str = "S",
        outcome: str = "Y",
        positions: Optional[dict[str, tuple[float, float]]] = None,
    ) -> str:
        """Render the DAG as a TikZ figure for LaTeX.

        Args:
            treatment: Treatment node name (highlighted).
            outcome: Outcome node name (highlighted).
            positions: Optional {node_name: (x, y)} layout coordinates.

        Returns:
            LaTeX/TikZ code string for the DAG figure.
        """
        if positions is None:
            positions = _default_layout(self)

        lines = [
            r"\begin{figure}[t]",
            r"\centering",
            r"\begin{tikzpicture}[",
            r"  >=stealth,",
            r"  node distance=2cm,",
            r"  observed/.style={circle, draw, thick, minimum size=1cm, font=\small},",
            r"  unobserved/.style={circle, draw, dashed, thick, minimum size=1cm, font=\small},",
            r"  treatment/.style={circle, draw, very thick, fill=blue!10, minimum size=1cm, font=\small},",
            r"  outcome/.style={circle, draw, very thick, fill=red!10, minimum size=1cm, font=\small},",
            r"  every edge/.style={draw, ->, thick},",
            r"]",
        ]

        # Nodes
        for name, node in self.nodes.items():
            x, y = positions.get(name, (0, 0))
            if name == treatment:
                style = "treatment"
            elif name == outcome:
                style = "outcome"
            elif not node.observed:
                style = "unobserved"
            else:
                style = "observed"
            lines.append(
                f"  \\node[{style}] ({name}) at ({x},{y}) {{{node.label}}};"
            )

        lines.append("")

        # Edges
        for src, tgt in self.edges:
            style = "dashed" if not self.nodes[src].observed else ""
            edge_opts = f"[{style}]" if style else ""
            lines.append(f"  \\draw[->] ({src}) edge{edge_opts} ({tgt});")

        lines += [
            r"\end{tikzpicture}",
            r"\caption{Assumed causal structure. "
            r"Solid circles are observed; dashed circles are unobserved. "
            r"The treatment node $S$ (sycophancy) is shaded blue; "
            r"the outcome $Y$ (win) is shaded red. "
            r"The path $S \to L \to Y$ represents mediation through length; "
            r"the path $U \to S$ and $U \to Y$ represents confounding by "
            r"unobserved model quality.}",
            r"\label{fig:dag}",
            r"\end{figure}",
        ]
        return "\n".join(lines)


def build_sycophancy_dag() -> CausalDAG:
    """Build the DAG for the sycophancy-preference analysis.

    Encodes the following assumed causal structure:
    - Model quality (U, unobserved) affects sycophancy, length, politeness, and winning
    - Prompt type (T) affects sycophancy, length, and winning
    - Sycophancy (S) affects length (mediator) and winning (direct effect)
    - Length (L) affects winning
    - Politeness (P) affects winning
    """
    dag = CausalDAG()

    dag.add_node("U", "U", observed=False,
                 description="Unobserved model quality / capability")
    dag.add_node("T", "T", observed=True,
                 description="Prompt type / domain")
    dag.add_node("S", "S", observed=True,
                 description="Sycophancy score")
    dag.add_node("L", "L", observed=True,
                 description="Response length")
    dag.add_node("P", "P", observed=True,
                 description="Politeness score")
    dag.add_node("Y", "Y", observed=True,
                 description="Win (Model A preferred)")

    # Unobserved model quality confounds everything
    dag.add_edge("U", "S")
    dag.add_edge("U", "L")
    dag.add_edge("U", "P")
    dag.add_edge("U", "Y")

    # Prompt type affects sycophancy, length, and outcome
    dag.add_edge("T", "S")
    dag.add_edge("T", "L")
    dag.add_edge("T", "Y")

    # Sycophancy -> Length (mediator path)
    dag.add_edge("S", "L")

    # Sycophancy -> Win (direct effect: the estimand)
    dag.add_edge("S", "Y")

    # Length -> Win (length bias)
    dag.add_edge("L", "Y")

    # Politeness -> Win
    dag.add_edge("P", "Y")

    return dag


def _default_layout(dag: CausalDAG) -> dict[str, tuple[float, float]]:
    """Default node positions for the sycophancy DAG."""
    return {
        "U": (3, 3),
        "T": (0, 1.5),
        "S": (2, 0),
        "L": (4, 0),
        "P": (0, 0),
        "Y": (6, 0),
    }


def _format_identification_reasoning(
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    mediators: set[str],
    unobserved: set[str],
) -> str:
    """Format the identification reasoning as prose."""
    parts = []
    parts.append(
        f"To identify the causal effect of {treatment} on {outcome}, "
        f"we apply the backdoor criterion."
    )
    if adjustment_set:
        parts.append(
            f"Conditioning on {{{', '.join(sorted(adjustment_set))}}} "
            f"blocks all backdoor (non-causal) paths from {treatment} to {outcome}."
        )
    if mediators:
        parts.append(
            f"Variables {{{', '.join(sorted(mediators))}}} lie on causal paths "
            f"from {treatment} to {outcome}. Conditioning on them estimates "
            f"the controlled direct effect; omitting them estimates the total effect."
        )
    if unobserved:
        parts.append(
            f"Unobserved variables {{{', '.join(sorted(unobserved))}}} "
            f"threaten identification. We assess robustness to this violation "
            f"using E-values and Rosenbaum sensitivity analysis."
        )
    return " ".join(parts)
