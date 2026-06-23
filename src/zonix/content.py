from __future__ import annotations

from typing import Any


def content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return [{"type": "text", "text": str(content)}]


def content_text(content: Any, *, include_images: bool = False) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content_blocks(content):
        block_type = str(block.get("type") or "").strip()
        if block_type in {"text", "input_text"}:
            text = block.get("text", block.get("content"))
            if isinstance(text, str):
                parts.append(text)
            continue
        if block_type in {"image", "image_url", "input_image"} and include_images:
            filename = str(block.get("filename") or block.get("name") or "").strip()
            media_type = image_media_type(block) or "image"
            label = filename or media_type
            parts.append(f"[Image: {label}]")

    if parts:
        return "\n".join(part for part in parts if part)
    if isinstance(content, list | dict):
        return ""
    return str(content)


def has_image_content(content: Any) -> bool:
    return any(
        str(block.get("type") or "").strip() in {"image", "image_url", "input_image"}
        for block in content_blocks(content)
    )


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(
    image: str,
    *,
    media_type: str | None = None,
    filename: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "image", "image": image}
    if media_type:
        block["media_type"] = media_type
    if filename:
        block["filename"] = filename
    if detail:
        block["detail"] = detail
    return block


def image_source(block: dict[str, Any]) -> str:
    raw = block.get("image", block.get("url", block.get("data_url")))
    if isinstance(raw, str) and raw:
        return raw
    image_url = block.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    return ""


def image_detail(block: dict[str, Any]) -> str | None:
    detail = block.get("detail")
    return detail if isinstance(detail, str) and detail else None


def image_media_type(block: dict[str, Any]) -> str | None:
    for key in ("media_type", "mediaType", "mime_type", "mimeType"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    source = image_source(block)
    media_type, _ = split_data_url(source)
    return media_type


def split_data_url(value: str) -> tuple[str | None, str | None]:
    if not value.startswith("data:"):
        return None, None
    header, separator, data = value.partition(",")
    if not separator:
        return None, None
    media_type = header.removeprefix("data:").split(";", 1)[0] or None
    return media_type, data
