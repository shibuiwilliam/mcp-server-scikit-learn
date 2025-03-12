import base64
import binascii
import json


def is_base64(s: str) -> bool:
    try:
        decoded_bytes = base64.b64decode(s, validate=True)
        return base64.b64encode(decoded_bytes).decode("utf-8") == s
    except binascii.Error:
        return False
    except UnicodeDecodeError:
        return False


def string_to_list_dict(s: str) -> list[dict[str, str | int | float | bool | None]]:
    ss = s
    if is_base64(s):
        ss = base64.b64decode(s).decode("utf-8")
    try:
        return json.loads(ss)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")
