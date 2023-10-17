from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class CommonScreensTable(Base):
    __tablename__ = "common_screens"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_url: Mapped[str] = mapped_column(String(30))
    image_url: Mapped[Optional[str]]
    created_at: Mapped[Optional[str]]
    website_ip: Mapped[Optional[str]]
    domain: Mapped[Optional[str]]
    category: Mapped[Optional[str]]
    size: Mapped[Optional[str]]
    country_code: Mapped[Optional[str]]
    page_lang: Mapped[Optional[str]]
    harmonicc_pos: Mapped[Optional[str]]
    pr_pos: Mapped[Optional[str]]
    n_hosts: Mapped[Optional[str]]
    ptile: Mapped[Optional[str]]
    title: Mapped[Optional[str]]
    description: Mapped[Optional[str]]
    keywords: Mapped[Optional[str]]

    def __repr__(self) -> str:
        return f"CommonScreensTable(id={self.id!r}, site_url={self.site_url!r}, image_url={self.image_url!r})"



