from sqlalchemy import Column, String, Float, DateTime, func
import uuid

from ..database import Base

class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    language = Column(String, nullable=False, default="en")
    category = Column(String, nullable=False)
    material = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    details = Column(String, nullable=False)  # JSON string of full result
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

