"""SQLAlchemy ORM models for the Crown data store.

Entities:
  Applicant            -- raw + normalized applicant data and underwriting outcome
  Account              -- an approved, opened card account (1:1 with an approved applicant)
  CreditLimitStepUp    -- tenure-based limit schedule rows per account
  Transaction          -- individual card purchases with merchant category
  Statement            -- monthly billing cycle rollup
  CourseCompletion     -- financial-education course completion record
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..config import db_url


class Base(DeclarativeBase):
    pass


class Applicant(Base):
    __tablename__ = "applicants"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    school_year: Mapped[str] = mapped_column(String(20))
    monthly_income: Mapped[float] = mapped_column(Float)   # income/allowance, USD
    is_thin_file: Mapped[bool] = mapped_column(Boolean)
    credit_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dti: Mapped[float] = mapped_column(Float)
    enrolled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Underwriting outcome
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    decline_reason: Mapped[str | None] = mapped_column(String(120), nullable=True)
    assigned_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    applied_at: Mapped[date] = mapped_column(Date, default=date.today)

    account: Mapped["Account | None"] = relationship(back_populates="applicant", uselist=False)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_id: Mapped[int] = mapped_column(ForeignKey("applicants.id"))
    opened_at: Mapped[date] = mapped_column(Date)
    starter_limit: Mapped[float] = mapped_column(Float)
    current_limit: Mapped[float] = mapped_column(Float)
    is_thin_file: Mapped[bool] = mapped_column(Boolean)
    school_year: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|churned|charged_off

    applicant: Mapped[Applicant] = relationship(back_populates="account")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="account")
    statements: Mapped[list["Statement"]] = relationship(back_populates="account")
    step_ups: Mapped[list["CreditLimitStepUp"]] = relationship(back_populates="account")
    course: Mapped["CourseCompletion | None"] = relationship(
        back_populates="account", uselist=False
    )


class CreditLimitStepUp(Base):
    __tablename__ = "credit_limit_step_ups"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    effective_month: Mapped[int] = mapped_column(Integer)  # months since open
    limit: Mapped[float] = mapped_column(Float)

    account: Mapped[Account] = relationship(back_populates="step_ups")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    posted_at: Mapped[date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Float)
    merchant_category: Mapped[str] = mapped_column(String(40))
    is_partner_merchant: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped[Account] = relationship(back_populates="transactions")


class Statement(Base):
    __tablename__ = "statements"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    cycle_month: Mapped[int] = mapped_column(Integer)   # 1-indexed months since open
    period_end: Mapped[date] = mapped_column(Date)
    purchases: Mapped[float] = mapped_column(Float)
    revolving_balance: Mapped[float] = mapped_column(Float)
    interest_charged: Mapped[float] = mapped_column(Float, default=0.0)
    late_fee_charged: Mapped[float] = mapped_column(Float, default=0.0)
    in_grace: Mapped[bool] = mapped_column(Boolean, default=False)

    account: Mapped[Account] = relationship(back_populates="statements")


class CourseCompletion(Base):
    __tablename__ = "course_completions"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    account: Mapped[Account] = relationship(back_populates="course")


def get_engine(path=None, echo: bool = False):
    return create_engine(db_url(path), echo=echo, future=True)


def get_sessionmaker(path=None, echo: bool = False):
    return sessionmaker(bind=get_engine(path, echo=echo), future=True)


def init_db(path=None, drop: bool = False):
    """Create (optionally reset) all tables. Returns the engine."""
    engine = get_engine(path)
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine
