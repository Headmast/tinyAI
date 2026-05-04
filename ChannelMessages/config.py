from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()


def _get(name: str, alt: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if (val is None or val == "") and alt:
        val = os.getenv(alt)
    return val if (val is not None and val != "") else default


API_ID: int = int(_get("API_ID", "APIID", "0"))
API_HASH: str = _get("API_HASH", "APIHASH", "")
PHONE_NUMBER: str = _get("PHONE_NUMBER", default="")
SESSION_NAME: str = _get("SESSION_NAME", "SESSIONNAME", "channel_stats")
DB_URL: str = _get("DB_URL", "DBURL", "sqlite+aiosqlite:///channel_stats.db")

# Number of recent messages to fetch per channel
MESSAGES_LIMIT: int = int(_get("MESSAGES_LIMIT", default="10"))
