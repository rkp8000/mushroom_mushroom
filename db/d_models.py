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

    f_behav = Column(String)
    f_gcamp = Column(String)
    f_t_gcamp = Column(String)
    f_light = Column(String)
    f_air = Column(String)
    f_odor_binary = Column(String)
    f_odor_pid = Column(String)
    
    pfx_clean = Column(String)
    
    details = Column(JSONB)
    
    walking_threshold = Column(Float)
