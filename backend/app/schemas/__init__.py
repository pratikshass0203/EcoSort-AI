from pydantic import BaseModel, Field
from typing import List, Optional

class ScanOut(BaseModel):
    id: str
    filename: str
    language: str
    category: str
    material: str
    confidence: float
    details: dict
    timestamp: str

    class Config:
        orm_mode = True

class CategoryItem(BaseModel):
    id: int
    name: str

class CategoryList(BaseModel):
    categories: List[CategoryItem]

class AssistantMessage(BaseModel):
    message: str
    language: str = Field(default="en", pattern="^(en|hi|ta)$")

class SettingOut(BaseModel):
    key: str
    value: str
    description: Optional[str] = None
    mutable: bool = True

    class Config:
        orm_mode = True

