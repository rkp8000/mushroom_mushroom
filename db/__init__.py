"""
Class for reading from and writing to database.
"""
from datetime import datetime
from getpass import getpass
import os
from pprint import pprint

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .d_models import Base
import LOCAL as L


def make_session():
    """
    Connect to the database and return a new session object for that database.
    
    :return: session object
    """

    # build connection url from input
    try:
        user = L.POSTGRES_USER
        pw = L.POSTGRES_PW
        db_name = L.POSTGRES_DB_NAME
    except:
        raise NameError(
            'Username, password, and db_name must be '
            ' specified in "LOCAL_SETTINGS.py".')

    url = 'postgres://{}:{}@/{}'.format(user, pw, db_name)

    # make and connect an engine
    engine = create_engine(url)
    engine.connect()

    # create all tables defined in d_models.py
    Base.metadata.create_all(engine)

    # get a new session
    session = sessionmaker(bind=engine)()

    return session


def check_tables_not_empty(session, *models):
    """
    Check whether a list of tables in the db are empty.

    :param session: session
    :param models: list of database models
    :return: list of names of tables that are empty
    """

    empty_table_list = []

    for model in models:

        if session.query(model).count() == 0:
            empty_table_list.append(model.__tablename__)

    if empty_table_list:

        prefix = 'The following tables must be populated before calling this function:'
        empty_table_string = ', '.join(empty_table_list)

        empty_table_message = '{} {}'.format(prefix, empty_table_string)

        raise Exception(empty_table_message)


def empty_tables(session, *models):
    """
    Empty a list of tables.

    TODO: make sure deletion cascade works properly:

    http://stackoverflow.com/questions/5033547/sqlachemy-cascade-delete

    :param session: session
    :param models: list of database models
    """
    for model in models:
        session.query(model).delete()
        session.commit()


def delete_record_group(session, group_field, group_name):
    """
    Delete group of records from a model.
    :param session: session instance
    :param group_field: model field corresponding to group
    :param group_name: name of group to delete
    """

    session.query(group_field.class_).filter(group_field == group_name).delete()
    session.commit()
