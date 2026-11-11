from sqlalchemy import Column, String, Boolean
from ..database import Base

class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
    description = Column(String, nullable=True)
    mutable = Column(Boolean, default=True)

