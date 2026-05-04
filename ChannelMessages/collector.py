"""
collector.py — fetches last N messages from every channel the user is in,
computes per-channel statistics, and persists everything to the database.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import Counter
from datetime import datetime, timezone
from getpass import getpass
from typing import Optional

from sqlalchemy import select
from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    SessionPasswordNeededError,
)
from telethon.tl.types import Channel as TLChannel, User as TLUser

try:
    import qrcode
except ImportError:
    qrcode = None

from config import API_ID, API_HASH, PHONE_NUMBER, SESSION_NAME, MESSAGES_LIMIT
from db import SessionLocal
from models import Channel, ChannelStats, Message


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sender_display_name(sender: Optional[object]) -> str:
    """Return a human-readable display name for a Telegram entity."""
    if sender is None:
        return "Unknown"
    if isinstance(sender, TLUser):
        parts = filter(None, [sender.first_name, sender.last_name])
        name = " ".join(parts).strip()
        return name or sender.username or str(sender.id)
    if hasattr(sender, "title"):
        return sender.title or str(getattr(sender, "id", ""))
    return str(getattr(sender, "id", "Unknown"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _delivery_hint(sent_code: object) -> str:
    """Explain where Telegram delivered the login code."""
    sent_type = getattr(sent_code, "type", None)
    sent_type_name = type(sent_type).__name__ if sent_type else "Unknown"

    if sent_type_name == "SentCodeTypeApp":
        return (
            "Telegram sent the login code to an already signed-in Telegram app. "
            "Check Telegram on your phone or desktop; this is not an SMS."
        )
    if sent_type_name == "SentCodeTypeSms":
        return "Telegram sent the login code by SMS to your phone number."
    if sent_type_name == "SentCodeTypeCall":
        return "Telegram will deliver the login code by phone call."
    if sent_type_name == "SentCodeTypeFlashCall":
        return "Telegram will deliver the login code by flash call."
    if sent_type_name == "SentCodeTypeMissedCall":
        return "Telegram will deliver the login code through a missed call pattern."

    return f"Telegram requested login confirmation. Delivery type: {sent_type_name}."


async def _ensure_authorized(client: TelegramClient) -> None:
    """Sign in the user if the current session is not authorized yet."""
    logger.info("Connecting Telegram client")
    await client.connect()

    if await client.is_user_authorized():
        logger.info("Telegram session is already authorized")
        return

    logger.info("Starting QR login via tg://login?token= URL")
    qr_login = await client.qr_login()

    # Encode token bytes to base64url (no padding) — standard Telegram QR URL format
    token_b64 = base64.urlsafe_b64encode(qr_login.token).rstrip(b"=").decode()
    tg_url = f"tg://login?token={token_b64}"

    print("\n" + "=" * 70)
    print("TELEGRAM LOGIN")
    print("=" * 70)
    print(f"\n🔗 Telegram link:\n")
    print(f"   {tg_url}\n")

    if qrcode is not None:
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=2,
            )
            qr.add_data(tg_url)
            qr.make(fit=True)
            qr_image_path = "telegram_qr_code.jpg"
            qr.make_image(fill_color="black", back_color="white").save(qr_image_path, "JPEG")
            print(f"✓ QR code saved: {qr_image_path}")
            print("  Open this file and scan it with Telegram\n")
        except Exception as exc:
            logger.warning("Could not create QR image: %s", exc)
    else:
        print("Install qrcode library to generate JPEG: pip install qrcode[pil]\n")

    print("How to login:")
    print("  Mobile: tap the link above to open Telegram")
    print("  Desktop: open telegram_qr_code.jpg and scan it with Telegram on your phone")
    print("=" * 70 + "\n")

    try:
        await qr_login.wait(timeout=120)
        logger.info("QR login successful")
        print("\n✓ Login successful!")
    except asyncio.TimeoutError:
        logger.error("QR login timeout (120s). Run the script again.")
        raise



# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

async def collect() -> list[dict]:
    """
    Connect as the user, iterate all channels, collect MESSAGES_LIMIT recent
    messages per channel, compute stats, persist to DB.

    Returns a list of summary dicts for display.
    """
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    await _ensure_authorized(client)

    summaries: list[dict] = []

    async with client:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity

            # Only process channels (broadcast channels, not regular groups)
            if not isinstance(entity, TLChannel):
                continue
            if entity.megagroup:
                # Skip supergroups / megagroups; keep only broadcast channels
                continue

            channel_tg_id: int = entity.id
            title: str = entity.title or ""
            username: Optional[str] = getattr(entity, "username", None)

            # --- Upsert channel row ---
            async with SessionLocal() as db:
                result = await db.execute(
                    select(Channel).where(Channel.tg_id == channel_tg_id)
                )
                channel_row: Optional[Channel] = result.scalar_one_or_none()

                if channel_row is None:
                    channel_row = Channel(
                        tg_id=channel_tg_id,
                        title=title,
                        username=username,
                    )
                    db.add(channel_row)
                    await db.flush()
                else:
                    channel_row.title = title
                    channel_row.username = username

                # --- Fetch last N messages ---
                sender_counter: Counter[tuple[Optional[int], str]] = Counter()
                media_count = 0
                total_chars = 0
                fetched_count = 0

                async for msg in client.iter_messages(entity, limit=MESSAGES_LIMIT):
                    # Resolve sender info
                    sender = await msg.get_sender()
                    sender_id: Optional[int] = getattr(sender, "id", None)
                    sender_name: str = _sender_display_name(sender)

                    text: str = msg.text or ""
                    has_media: bool = msg.media is not None
                    char_len: int = len(text)

                    total_chars += char_len
                    if has_media:
                        media_count += 1
                    sender_counter[(sender_id, sender_name)] += 1
                    fetched_count += 1

                    # Upsert message row
                    msg_result = await db.execute(
                        select(Message).where(
                            Message.channel_tg_id == channel_tg_id,
                            Message.msg_id == msg.id,
                        )
                    )
                    existing_msg: Optional[Message] = msg_result.scalar_one_or_none()

                    if existing_msg is None:
                        db.add(
                            Message(
                                channel_tg_id=channel_tg_id,
                                msg_id=msg.id,
                                date=msg.date.replace(tzinfo=None) if msg.date else None,
                                sender_id=sender_id,
                                sender_name=sender_name,
                                text=text,
                                has_media=has_media,
                                char_length=char_len,
                            )
                        )
                    else:
                        # Refresh in case the message was edited
                        existing_msg.text = text
                        existing_msg.has_media = has_media
                        existing_msg.char_length = char_len

                # --- Compute stats ---
                avg_chars = (total_chars / fetched_count) if fetched_count else 0.0
                top_key = sender_counter.most_common(1)[0][0] if sender_counter else (None, "")
                top_sender_id, top_sender_name = top_key

                collected_at = _utc_now()

                db.add(
                    ChannelStats(
                        channel_tg_id=channel_tg_id,
                        collected_at=collected_at,
                        msg_count=fetched_count,
                        avg_char_length=round(avg_chars, 2),
                        media_count=media_count,
                        top_sender_id=top_sender_id,
                        top_sender_name=top_sender_name,
                    )
                )

                channel_row.last_collected_at = collected_at
                await db.commit()

            summaries.append(
                {
                    "title": title,
                    "username": username,
                    "msg_count": fetched_count,
                    "avg_chars": round(avg_chars, 1),
                    "media_count": media_count,
                    "top_sender": top_sender_name,
                }
            )

    return summaries
