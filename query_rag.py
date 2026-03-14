from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = ROOT / "generated" / "graphrag-documents.json"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "last-rag-query.json"
EMBED_MODEL = "gemini-embedding-001"
KEY_PREFIX = "frontend-rag:"


def get_env(name: str) -> str:
    value = os.getenv(name, "").strip().strip("\"'")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def infer_region(index_arn: str) -> str | None:
    parts = index_arn.split(":")
    if len(parts) >= 4 and parts[2] == "s3vectors":
        return parts[3]
    return None


def strip_prefix(key: str) -> str:
    return key[len(KEY_PREFIX) :] if key.startswith(KEY_PREFIX) else key


class GraphRAGRetriever:
    def __init__(
        self,
        docs_path: Path,
        index_arn: str,
        aws_region: str,
        google_api_key: str,
    ) -> None:
        payload = json.loads(docs_path.read_text(encoding="utf-8"))
        self.documents: list[dict[str, Any]] = payload.get("documents", [])
        self.docs_by_key = {doc["key"]: doc for doc in self.documents}
        self.docs_by_source_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.relations_by_source_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.bundles_by_feature_area: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for doc in self.documents:
            metadata = doc.get("metadata", {})
            source_path = metadata.get("sourcePath")
            feature_area = metadata.get("featureArea")
            if source_path:
                self.docs_by_source_path[source_path].append(doc)
            if doc.get("type") == "relation" and source_path:
                self.relations_by_source_path[source_path].append(doc)
            if doc.get("type") == "feature_bundle" and feature_area:
                self.bundles_by_feature_area[feature_area].append(doc)

        self.genai_client = genai.Client(api_key=google_api_key)
        self.s3vectors = boto3.client("s3vectors", region_name=aws_region)
        self.index_arn = index_arn

    def _embed_query(self, text: str) -> list[float]:
        result = self.genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        if not result.embeddings or not result.embeddings[0].values:
            raise RuntimeError("Embedding API returned no vector for the query.")
        return list(result.embeddings[0].values)

    def _query_keys(self, text: str, top_k: int) -> list[dict[str, Any]]:
        vector = self._embed_query(text)
        response = self.s3vectors.query_vectors(
            indexArn=self.index_arn,
            queryVector={"float32": vector},
            topK=top_k,
            returnMetadata=False,
        )
        return response.get("vectors", [])

    def retrieve(
        self,
        text: str,
        *,
        top_k: int = 30,
        max_components: int = 8,
        max_relations: int = 20,
        max_bundles: int = 5,
    ) -> dict[str, Any]:
        vector_hits = self._query_keys(text, top_k=top_k)

        resolved_hits: list[dict[str, Any]] = []
        for hit in vector_hits:
            key = hit.get("key")
            if not key:
                continue
            doc = self.docs_by_key.get(strip_prefix(key))
            if not doc:
                continue
            resolved_hits.append(
                {
                    "key": doc["key"],
                    "type": doc["type"],
                    "doc": doc,
                    "score": hit.get("score"),
                    "distance": hit.get("distance"),
                }
            )

        primary_components: list[dict[str, Any]] = []
        primary_relations: list[dict[str, Any]] = []
        primary_bundles: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for item in resolved_hits:
            doc = item["doc"]
            key = doc["key"]
            if key in seen_keys:
                continue
            if doc["type"] == "module_component" and len(primary_components) < max_components:
                primary_components.append(item)
                seen_keys.add(key)
            elif doc["type"] == "relation" and len(primary_relations) < max_relations:
                primary_relations.append(item)
                seen_keys.add(key)
            elif doc["type"] == "feature_bundle" and len(primary_bundles) < max_bundles:
                primary_bundles.append(item)
                seen_keys.add(key)

        component_source_paths = {
            item["doc"]["metadata"]["sourcePath"]
            for item in primary_components
            if item["doc"].get("metadata", {}).get("sourcePath")
        }
        feature_areas = {
            item["doc"]["metadata"]["featureArea"]
            for item in primary_components
            if item["doc"].get("metadata", {}).get("featureArea")
        }
        feature_areas.update(
            item["doc"]["metadata"]["featureArea"]
            for item in primary_relations
            if item["doc"].get("metadata", {}).get("featureArea")
        )

        expanded_relations: list[dict[str, Any]] = []
        seen_relation_keys = {item["doc"]["key"] for item in primary_relations}
        for source_path in component_source_paths:
            for relation_doc in self.relations_by_source_path.get(source_path, []):
                if relation_doc["key"] in seen_relation_keys:
                    continue
                expanded_relations.append(relation_doc)
                seen_relation_keys.add(relation_doc["key"])
                if len(expanded_relations) >= max_relations:
                    break
            if len(expanded_relations) >= max_relations:
                break

        expanded_bundles: list[dict[str, Any]] = []
        seen_bundle_keys = {item["doc"]["key"] for item in primary_bundles}
        for feature_area in feature_areas:
            for bundle_doc in self.bundles_by_feature_area.get(feature_area, []):
                if bundle_doc["key"] in seen_bundle_keys:
                    continue
                expanded_bundles.append(bundle_doc)
                seen_bundle_keys.add(bundle_doc["key"])
                if len(expanded_bundles) >= max_bundles:
                    break
            if len(expanded_bundles) >= max_bundles:
                break

        expanded_components: list[dict[str, Any]] = []
        seen_component_keys = {item["doc"]["key"] for item in primary_components}
        for relation_doc in expanded_relations[:max_relations]:
            target_path = relation_doc.get("metadata", {}).get("targetPath")
            if not target_path:
                continue
            for candidate in self.docs_by_source_path.get(target_path, []):
                if candidate["type"] != "module_component":
                    continue
                if candidate["key"] in seen_component_keys:
                    continue
                expanded_components.append(candidate)
                seen_component_keys.add(candidate["key"])
                if len(expanded_components) >= max_components:
                    break
            if len(expanded_components) >= max_components:
                break

        return {
            "query": text,
            "topK": top_k,
            "primary": {
                "components": primary_components,
                "relations": primary_relations,
                "bundles": primary_bundles,
            },
            "expanded": {
                "components": expanded_components,
                "relations": expanded_relations,
                "bundles": expanded_bundles,
            },
            "resolvedHitCount": len(resolved_hits),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query GraphRAG over local JSON + S3 Vectors.")
    parser.add_argument("query", help="Natural-language query to retrieve relevant GraphRAG context for.")
    parser.add_argument("--top-k", type=int, default=30, help="How many vector hits to request from S3 Vectors.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the JSON retrieval result.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    google_api_key = get_env("GOOGLE_API_KEY")
    index_arn = get_env("AWS_S3_VECTOR_INDEX_ARN")
    aws_region = os.getenv("AWS_REGION", "").strip() or infer_region(index_arn)
    if not aws_region:
        raise RuntimeError(
            "Missing AWS_REGION and could not infer the AWS region from AWS_S3_VECTOR_INDEX_ARN."
        )

    retriever = GraphRAGRetriever(
        docs_path=DOCUMENTS_PATH,
        index_arn=index_arn,
        aws_region=aws_region,
        google_api_key=google_api_key,
    )
    result = retriever.retrieve(args.query, top_k=args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Resolved {result['resolvedHitCount']} vector hit(s).")
    print(f"Primary components: {len(result['primary']['components'])}")
    print(f"Expanded relations: {len(result['expanded']['relations'])}")
    print(f"Wrote retrieval result to {output_path}")


if __name__ == "__main__":
    main()
