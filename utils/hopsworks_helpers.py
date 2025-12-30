"""
Hopsworks connection and helper utilities.
"""

import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()


def get_hopsworks_project():
    """
    Connect to Hopsworks project.

    Returns:
        Hopsworks project object
    """
    api_key = os.getenv('HOPSWORKS_API_KEY')
    project_name = os.getenv('HOPSWORKS_PROJECT_NAME')

    if not api_key or not project_name:
        raise ValueError("HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME must be set in .env")

    project = hopsworks.login(api_key_value=api_key, project=project_name)

    return project


def get_feature_store(project=None):
    """
    Get feature store from Hopsworks project.

    Args:
        project: Hopsworks project (if None, will connect)

    Returns:
        Feature store object
    """
    if project is None:
        project = get_hopsworks_project()

    fs = project.get_feature_store()

    return fs


def create_feature_group(fs, name: str, df, primary_key: list, description: str = "", version: int = 1):
    """
    Create or get feature group in Hopsworks.

    Args:
        fs: Feature store object
        name: Feature group name
        df: DataFrame with features
        primary_key: List of primary key columns
        description: Feature group description
        version: Version number

    Returns:
        Feature group object
    """
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time='date'
    )

    fg.insert(df, overwrite=True)

    return fg


def get_model_registry(project=None):
    """
    Get model registry from Hopsworks project.

    Args:
        project: Hopsworks project (if None, will connect)

    Returns:
        Model registry object
    """
    if project is None:
        project = get_hopsworks_project()

    mr = project.get_model_registry()

    return mr
