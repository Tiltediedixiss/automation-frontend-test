from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "generated"
OUTPUT_PATH = OUTPUT_DIR / "graphrag-documents.json"

DEFAULT_SOURCE = ROOT / "our_components.json"


def _source_file_str(source_path: Path) -> str:
    try:
        return str(source_path.relative_to(ROOT))
    except ValueError:
        return str(source_path)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def normalize_whitespace(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_feature_area(source_path: str) -> str:
    if source_path.startswith("src/components/ui/"):
        return "global-ui"
    if source_path.startswith("src/app/shells/"):
        return "shells"

    parts = source_path.split("/")
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "components":
        return parts[2]
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "app":
        return f"app-{parts[2]}"
    return parts[1] if len(parts) > 1 else "misc"


def infer_role(feature_area: str, source_path: str) -> str:
    lower = f"{feature_area} {source_path}".lower()
    if "student" in lower:
        return "student"
    if "teacher" in lower:
        return "teacher"
    if "principal" in lower:
        return "principal"
    if "admin" in lower:
        return "admin"
    if "auth" in lower:
        return "auth"
    return "shared"


def infer_component_kind(source_path: str, symbol_name: str, description: str) -> str:
    lower = f"{source_path} {symbol_name} {description}".lower()
    if "shell" in lower:
        return "shell"
    if "modal" in lower or "dialog" in lower:
        return "modal"
    if "form" in lower:
        return "form"
    if "sidebar" in lower:
        return "sidebar"
    if "button" in lower:
        return "button"
    if "card" in lower:
        return "card"
    if "dashboard" in lower:
        return "dashboard"
    if "page" in lower:
        return "page"
    if "layout" in lower:
        return "layout"
    return "component"


def flatten_imports(imports_value: Any) -> list[str]:
    if not imports_value:
        return []
    if isinstance(imports_value, list):
        return [str(item) for item in imports_value if item]
    if isinstance(imports_value, dict):
        flattened: list[str] = []
        for values in imports_value.values():
            if isinstance(values, list):
                flattened.extend(str(item) for item in values if item)
        return flattened
    return []


def summarize_import_groups(imports_value: Any) -> str:
    if not imports_value:
        return ""
    if isinstance(imports_value, list):
        return ", ".join(flatten_imports(imports_value))
    if isinstance(imports_value, dict):
        parts = []
        for group, items in imports_value.items():
            if isinstance(items, list) and items:
                parts.append(f"{group}: {', '.join(str(item) for item in items)}")
        return " | ".join(parts)
    return ""


def extract_state_count(summary_text: str) -> int:
    match = re.search(r"(\d+)\s+useState", summary_text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def build_module_component_documents(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for component in components:
        component_path = component["path"]
        feature_area = infer_feature_area(component_path)
        role = infer_role(feature_area, component_path)
        symbol_name = component_path.split("#")[-1]
        imports = flatten_imports(component.get("imports"))
        output_text = normalize_whitespace(component.get("output", ""))
        description_text = normalize_whitespace(component.get("description", ""))
        tags = [
            "module-component",
            feature_area,
            role,
            infer_component_kind(component_path, symbol_name, description_text),
        ]
        if "route navigation" in output_text:
            tags.append("navigation")
        if "useauthstore" in description_text.lower():
            tags.append("auth-store")
        if "loads data with" in description_text.lower():
            tags.append("query")
        if "updates data with" in description_text.lower():
            tags.append("mutation")

        docs.append(
            {
                "key": f"component:{component_path}",
                "type": "module_component",
                "text": normalize_whitespace(
                    f"""
                    Frontend module component.
                    Path: {component_path}
                    Symbol: {symbol_name}
                    Feature area: {feature_area}
                    Role: {role}
                    Input signature: {component.get("input", "")}
                    Behavior summary: {output_text}
                    Description: {description_text}
                    Local state slots: {extract_state_count(output_text)}
                    Imported dependencies: {", ".join(imports) or "none"}
                    """
                ),
                "metadata": {
                    "sourcePath": component_path,
                    "featureArea": feature_area,
                    "role": role,
                    "componentKind": infer_component_kind(component_path, symbol_name, description_text),
                    "importCount": len(imports),
                    "localStateCount": extract_state_count(output_text),
                    "tags": tags,
                },
            }
        )
    return docs


def build_relation_documents(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for component in components:
        source = component["path"]
        feature_area = infer_feature_area(source)
        for target in flatten_imports(component.get("imports")):
            docs.append(
                {
                    "key": f"edge:{source}->{target}",
                    "type": "relation",
                    "text": normalize_whitespace(
                        f"""
                        Frontend dependency relationship.
                        Source component: {source}
                        Imported dependency: {target}
                        Feature area: {feature_area}
                        Relationship: the source component composes or depends on the imported symbol while implementing its UI behavior.
                        """
                    ),
                    "metadata": {
                        "sourcePath": source,
                        "targetPath": target,
                        "featureArea": feature_area,
                        "tags": ["relation", feature_area],
                    },
                }
            )
    return docs


def build_bundle_documents(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        grouped[infer_feature_area(component["path"])].append(component)

    docs = []
    for feature_area, items in grouped.items():
        role = infer_role(feature_area, items[0]["path"])
        symbols = [item["path"] for item in items][:40]
        flattened_imports: list[str] = []
        for item in items:
            flattened_imports.extend(flatten_imports(item.get("imports")))

        docs.append(
            {
                "key": f"bundle:{feature_area}",
                "type": "feature_bundle",
                "text": normalize_whitespace(
                    f"""
                    Frontend feature bundle for retrieval expansion.
                    Feature area: {feature_area}
                    Role: {role}
                    Member count: {len(items)}
                    Important symbols: {", ".join(symbols)}
                    Common dependencies: {", ".join(dict.fromkeys(flattened_imports)) or "none"}
                    """
                ),
                "metadata": {
                    "featureArea": feature_area,
                    "role": role,
                    "memberCount": len(items),
                    "tags": ["bundle", feature_area, role],
                },
            }
        )
    return docs


def main() -> None:
    source_path = DEFAULT_SOURCE
    if len(sys.argv) > 1:
        source_path = Path(sys.argv[1]).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Components file not found: {source_path}")

    module_components = json.loads(source_path.read_text(encoding="utf-8"))

    documents = [
        *build_module_component_documents(module_components),
        *build_relation_documents(module_components),
        *build_bundle_documents(module_components),
    ]

    counts = Counter(doc["type"] for doc in documents)
    payload = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "sourceFiles": [_source_file_str(source_path)],
        "counts": dict(counts),
        "documents": documents,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {len(documents)} GraphRAG documents from {source_path.name} -> {OUTPUT_PATH.relative_to(ROOT)}")
    print(dict(counts))


if __name__ == "__main__":
    main()
