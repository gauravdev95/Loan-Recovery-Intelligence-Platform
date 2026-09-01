"""Borrower browsing, dashboard stats and recovery queue."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.db import get_db
from backend.app.services import borrower_service

router = APIRouter(prefix="/api/v1", tags=["borrowers"])


@router.get("/borrowers")
def list_borrowers(
    search: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    min_probability: float | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    order: str = Query(default="desc"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows, total = borrower_service.list_borrowers(
        db,
        search=search,
        risk_level=risk_level,
        priority=priority,
        min_probability=min_probability,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
    )
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/borrowers/{borrower_id}")
def get_borrower(borrower_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = borrower_service.get_borrower(db, borrower_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Borrower not found.")
    return row


@router.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    return borrower_service.dashboard_stats(db)


@router.get("/recovery/priority")
def recovery_priority(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    queue = borrower_service.recovery_queue(db, limit=limit)
    return {"items": queue, "n": len(queue)}
