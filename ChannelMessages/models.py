from typing import Optional
from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)


class Base(DeclarativeBase):
    pass


class Channel(Base):
    """One row per Telegram channel the user is a member of."""
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    last_collected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Message(Base):
    """Last N messages fetched from each channel."""
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("channel_tg_id", "msg_id", name="uq_channel_msg"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_tg_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("channels.tg_id"), index=True)
    msg_id: Mapped[int] = mapped_column(Integer)
    date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sender_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    sender_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_media: Mapped[bool] = mapped_column(Boolean, default=False)
    char_length: Mapped[int] = mapped_column(Integer, default=0)


class ChannelStats(Base):
    """Aggregated statistics snapshot for a channel (one row per collection run)."""
    __tablename__ = "channel_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_tg_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("channels.tg_id"), index=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime)
    msg_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_char_length: Mapped[float] = mapped_column(Float, default=0.0)
    media_count: Mapped[int] = mapped_column(Integer, default=0)
    top_sender_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    top_sender_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
