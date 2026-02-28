from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    centroid: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    cluster_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cluster_description: Mapped[str | None] = mapped_column(String(400), nullable=True)
    cluster_keywords: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    items: Mapped[list["ContentItem"]] = relationship("ContentItem", back_populates="cluster")


class ContentItem(Base):
    __tablename__ = "content_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    original_input: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[int] = mapped_column(ForeignKey("clusters.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    cluster: Mapped[Cluster] = relationship("Cluster", back_populates="items")
