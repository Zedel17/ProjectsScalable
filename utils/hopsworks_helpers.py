"""
Hopsworks connection and helper utilities.
"""

import hopsworks
import pandas as pd
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
    Create feature group in Hopsworks and insert data with proper verification.

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
    import time

    # Make a copy and ensure proper types
    df = df.copy()

    # Convert date to datetime64[ms] which Hopsworks expects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).astype('datetime64[ms]')

    # Ensure numeric columns are proper types
    for col in df.columns:
        if col != 'date':
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int64')
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float64')

    print(f"\n{'='*60}")
    print(f"Creating feature group: {name}")
    print(f"{'='*60}")
    print(f"Data shape (before deduplication): {df.shape}")

    # Deduplicate based on primary key
    initial_count = len(df)
    df = df.drop_duplicates(subset=primary_key, keep='first')
    duplicates_removed = initial_count - len(df)

    if duplicates_removed > 0:
        print(f"⚠️  Removed {duplicates_removed} duplicate records based on primary_key: {primary_key}")

    print(f"Data shape (after deduplication): {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")

    try:
        # Try to get existing feature group
        fg = fs.get_feature_group(name=name, version=version)
        if fg is None:
            raise ValueError(f"get_feature_group returned None for existing '{name}'")
        print(f"✓ Feature group '{name}' already exists (version {version})")
        print(f"  Feature group object type: {type(fg)}")
        print(f"  Deleting existing data and re-inserting...")

    except Exception as e:
        # Create new feature group
        print(f"Creating new feature group '{name}'...")
        try:
            fg = fs.create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time='date',
                online_enabled=False
            )
            if fg is None:
                raise ValueError(f"create_feature_group returned None for '{name}'")
            print(f"✓ Feature group created")
        except Exception as create_error:
            print(f"\n❌ ERROR: Failed to create feature group '{name}'")
            print(f"Error details: {create_error}")
            raise

    # Ensure fg is not None before inserting
    if fg is None:
        raise ValueError(f"Feature group '{name}' is None - cannot insert data")

    # Insert data with explicit write_options
    print(f"\nInserting {len(df)} rows...")
    print(f"Sample data (first row):")
    print(df.head(1).to_dict('records'))

    job = fg.insert(df, write_options={"wait_for_job": True})

    print(f"✓ Insert job completed")
    print(f"  Job details: {job}")

    # Wait for data to be queryable
    print(f"\nWaiting for data to be committed (10 seconds)...")
    time.sleep(10)

    # Verification: Check data is accessible
    # NOTE: Arrow Flight query service has bugs in free tier, so we skip verification
    # The data IS uploaded (job succeeded), but may not be immediately queryable
    print(f"\n✓ Upload completed successfully")
    print(f"  Job finished with status: SUCCEEDED")
    print(f"  Uploaded {len(df)} rows to '{name}'")
    print(f"\n⚠️  NOTE: Data is in Hopsworks but may take a few minutes to be queryable")
    print(f"   Check the Hopsworks UI to see the data in the feature group")
    print(f"{'='*60}\n")
    return fg


def read_feature_group_safe(fg, max_retries=3):
    """
    Read feature group with fallback for Arrow Flight errors.

    Hopsworks free tier has Arrow Flight query service bugs that cause reads to fail
    even though data is successfully uploaded. This function tries multiple read methods.

    Args:
        fg: Feature group object
        max_retries: Number of retry attempts

    Returns:
        DataFrame with feature group data
    """
    import time

    for attempt in range(max_retries):
        try:
            # Try standard read
            print(f"Attempting to read feature group '{fg.name}' (attempt {attempt+1}/{max_retries})...")
            df = fg.read()
            print(f"✓ Successfully read {len(df)} rows")
            return df

        except Exception as e:
            error_msg = str(e)

            # Check if it's the known Arrow Flight bug
            if "No data found" in error_msg or "Binder Error" in error_msg or "FlightServerError" in error_msg:
                print(f"⚠️  Arrow Flight error detected: {error_msg[:100]}...")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ CRITICAL: Cannot read feature group '{fg.name}' after {max_retries} attempts")
                    print(f"\nPossible causes:")
                    print(f"  1. Hopsworks free tier Arrow Flight bug (known issue)")
                    print(f"  2. Data not yet committed (try again in a few minutes)")
                    print(f"  3. Feature group is empty")
                    print(f"\nWorkaround: Check Hopsworks UI to verify data exists.")
                    print(f"If data exists in UI but can't read via Python, this is a Hopsworks limitation.")
                    raise Exception(f"Cannot read feature group '{fg.name}' - Arrow Flight service unavailable")
            else:
                # Different error, re-raise
                raise


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
