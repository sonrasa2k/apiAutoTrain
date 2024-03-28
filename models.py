from sqlalchemy import Boolean, Column, ForeignKey, Integer, String,DateTime
from sqlalchemy.orm import relationship
from database import Base
import datetime
class Train(Base):
    __tablename__ = "Train"

    id = Column(String, primary_key=True)
    name_object  = Column(String,unique=False)
    id_model = Column(String,unique=False)
    name_moldel = Column(String,unique=False)


class Detect(Base):
    __tablename__ = "Detect"
    id = Column(String, primary_key=True)
    name_object = Column(String, unique=False)
    id_model = Column(String,unique=False)
    time_detect = Column(DateTime,default=datetime.datetime.utcnow)