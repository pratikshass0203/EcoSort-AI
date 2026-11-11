from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uuid
from typing import List

from ..database import get_db
from .. import schemas, models, services

router = APIRouter()

@router.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok"}

@router.post("/classify", summary="Classify waste image")
async def classify_image(
    file: UploadFile = File(...),
    language: str = Query("en", regex="^(en|hi|ta)$"),
    db: Session = Depends(get_db),
):
    # Read image bytes
    content = await file.read()
    # Perform classification (placeholder service)
    result = services.classify_image(content, language)
    # Store scan record
    scan = models.Scan(
        id=str(uuid.uuid4()),
        filename=file.filename,
        language=language,
        category=result["category"],
        material=result["material"],
        confidence=result["confidence"],
        details=result,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return schemas.ScanOut.from_orm(scan)

@router.get("/history", summary="Get scan history")
async def get_history(db: Session = Depends(get_db)):
    scans = db.query(models.Scan).order_by(models.Scan.timestamp.desc()).all()
    return [schemas.ScanOut.from_orm(s) for s in scans]

@router.get("/history/{scan_id}", summary="Get a specific scan")
async def get_scan(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(models.Scan).filter(models.Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return schemas.ScanOut.from_orm(scan)

@router.delete("/history/{scan_id}", summary="Delete a specific scan")
async def delete_scan(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(models.Scan).filter(models.Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    db.delete(scan)
    db.commit()
    return JSONResponse(content={"deleted": scan_id})

@router.get("/analytics", summary="Analytics based on scan records")
async def get_analytics(db: Session = Depends(get_db)):
    total = db.query(models.Scan).count()
    # Simple category count
    from sqlalchemy import func
    category_counts = (
        db.query(models.Scan.category, func.count(models.Scan.id))
        .group_by(models.Scan.category)
        .all()
    )
    return {"total_scans": total, "by_category": dict(category_counts)}

@router.get("/categories", summary="List waste categories")
async def list_categories():
    return schemas.CategoryList(categories=[
        {"id": 1, "name": "Biodegradable"},
        {"id": 2, "name": "Recyclable"},
        {"id": 3, "name": "E-waste"},
        {"id": 4, "name": "Hazardous"},
        {"id": 5, "name": "Unknown"},
    ])

@router.post("/assistant/chat", summary="Chat with local AI assistant")
async def assistant_chat(message: schemas.AssistantMessage, db: Session = Depends(get_db)):
    response = services.assistant_chat(message.message, message.language)
    return {"response": response}

@router.get("/device", summary="Get device information")
async def device_info():
    return services.get_device_info()

@router.get("/settings", summary="Get application settings")
async def get_settings(db: Session = Depends(get_db)):
    settings = db.query(models.Setting).all()
    return [schemas.SettingOut.from_orm(s) for s in settings]

