import numpy as np
from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, Integer, Float, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Trial(Base):

    __tablename__ = 'trial'

    id = Column(Integer, primary_key=True)
    
    name = Column(String, unique=True)
    expt = Column(String)
    fly = Column(String)
    number = Column(Integer)

    path = Column(String)

    file_name_behav = Column(String)
    file_name_gcamp = Column(String)
    file_name_gcamp_timestamp = Column(String)
    file_name_light_times = Column(String)
    file_name_air_tube = Column(String)
    
    details = Column(JSONB)
    
    walking_threshold = Column(Float)
