import os
import pandas as pd
from snowflake.connector import connect
from dotenv import load_dotenv

load_dotenv()

def get_snowflake_connection():
    account = os.getenv("SNOWFLAKE_ACCOUNT", "").strip()
    if not account:
        raise ValueError("SNOWFLAKE_ACCOUNT is not set. Check your .env file.")
    print(f"Connecting to Snowflake account: {account}")

    return connect(
        user=os.getenv("SNOWFLAKE_USER", "").strip(),
        password=os.getenv("SNOWFLAKE_PASSWORD", "").strip(),
        account=account,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE", "").strip(),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "").strip(),
    )

def run_query(query: str) -> pd.DataFrame:
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
    finally:
        cursor.close()
        conn.close()
    return df

def load_inpatient_claims() -> pd.DataFrame:
    query = "SELECT * FROM POC_TENNCARE.PUBLIC.INPATIENT_CLAIMS"
    return run_query(query)

def load_outpatient_claims() -> pd.DataFrame:
    query = "SELECT * FROM POC_TENNCARE.PUBLIC.OUTPATIENT_CLAIMS"
    return run_query(query)

def load_beneficiary_data() -> pd.DataFrame:
    query = "SELECT * FROM POC_TENNCARE.PUBLIC.BENEFICIARY_SUMMARY"
    return run_query(query)

def merge_claims_with_beneficiaries(inpatient_df, outpatient_df, beneficiary_df) -> pd.DataFrame:
    print("Joining inpatient, outpatient, and beneficiary data...")

    # Label claim types
    inpatient_df["CLAIM_TYPE"] = "inpatient"
    outpatient_df["CLAIM_TYPE"] = "outpatient"

    # Drop fully empty columns
    inpatient_df = inpatient_df.dropna(axis=1, how='all')
    outpatient_df = outpatient_df.dropna(axis=1, how='all')

    # Combine claims and create unique claim ID
    claims_df = pd.concat([inpatient_df, outpatient_df], ignore_index=True)
    claims_df["CLAIM_UID"] = claims_df["CLAIM_TYPE"].str.upper() + "_" + claims_df["CLM_ID"].astype(str)

    # ✅ Fix: Keep only one beneficiary row per DESYNPUF_ID (latest or first)
    beneficiary_df = beneficiary_df.sort_values("BENE_BIRTH_DT")  # optional: prefer oldest data
    beneficiary_df = beneficiary_df.drop_duplicates(subset="DESYNPUF_ID", keep="first")

    # Merge with deduplicated beneficiaries
    merged_df = claims_df.merge(beneficiary_df, on="DESYNPUF_ID", how="left")

    print(f"Merged claims total: {len(merged_df):,} rows")
    return merged_df

