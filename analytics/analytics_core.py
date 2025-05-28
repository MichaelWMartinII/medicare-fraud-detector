import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from llm.openai_client import ask_gpt

def engineer_core_features(df):
    print("Running core feature engineering...")

    df["CLM_ADMSN_DT"] = pd.to_datetime(df["CLM_ADMSN_DT"], errors="coerce")
    df["NCH_BENE_DSCHRG_DT"] = pd.to_datetime(df["NCH_BENE_DSCHRG_DT"], errors="coerce")

    df["LENGTH_OF_STAY"] = (df["NCH_BENE_DSCHRG_DT"] - df["CLM_ADMSN_DT"]).dt.days
    df["LENGTH_OF_STAY"] = df["LENGTH_OF_STAY"].fillna(0).clip(lower=0).astype(int)

    df["CLM_PMT_AMT"] = pd.to_numeric(df["CLM_PMT_AMT"], errors="coerce")
    df["COST_PER_DAY"] = df["CLM_PMT_AMT"] / df["LENGTH_OF_STAY"].replace(0, 1)

    return df


def flag_rules_with_progress(df):
    print("Applying refined rule-based flags...")

    tqdm.pandas(desc="⛏️ Flagging Rules")

    # Refined: Flag extremely high-cost 0-day stays
    df["RULE_ZERO_DAYS_HIGH_PAY"] = df.progress_apply(
        lambda row: (row["LENGTH_OF_STAY"] == 0) and (row["CLM_PMT_AMT"] > 25000), axis=1
    )

    # ✅ Fix: Use CLAIM_UID to avoid synthetic ID collision
    df["RULE_DUPLICATE_CLAIM_ID"] = df.duplicated(subset=["CLAIM_UID"], keep=False)

    # Apply combined rule flag
    df["IS_FLAGGED"] = (
        df["RULE_ZERO_DAYS_HIGH_PAY"] | df["RULE_DUPLICATE_CLAIM_ID"]
    ).map({True: "Y", False: "N"})

    # Reason column
    rule_descriptions = {
        "RULE_ZERO_DAYS_HIGH_PAY": "0-day stay with payment over $25k",
        "RULE_DUPLICATE_CLAIM_ID": "Duplicate claim UID"
    }

    df["REASON_FLAGGED"] = df.progress_apply(
        lambda row: "; ".join(desc for rule, desc in rule_descriptions.items() if row.get(rule)), axis=1
    )

    return df



def evaluate_predictions(df, predicted_column="IS_FLAGGED", label_column="IS_FLAGGED", print_report=True):
    y_true = df[label_column].map({"Y": 1, "N": 0})
    y_pred = df[predicted_column].map({"Y": 1, "N": 0})

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Flagged"], zero_division=0
    )

    if print_report:
        print("\nEvaluation Metrics")
        print(report)

    return report


def add_isolation_forest_scores(df):
    print("\nRunning Isolation Forest anomaly detection...")

    features = df[["CLM_PMT_AMT", "LENGTH_OF_STAY", "COST_PER_DAY"]].copy()
    features = features.fillna(0)

    model = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=0.01,
        random_state=42
    )
    df["IF_SCORE"] = model.fit_predict(features)
    df["IF_FLAGGED"] = (df["IF_SCORE"] == -1).map({True: "Y", False: "N"})

    return df


def summarize_isolation_forest_flags(df, debug=False, max_rows=None):
    flagged_df = df[df["IF_FLAGGED"] == "Y"].copy()
    print(f"\nGenerating AI summaries for {len(flagged_df)} Isolation Forest-flagged claims...")

    if max_rows is not None and len(flagged_df) > max_rows:
        flagged_df = flagged_df.head(max_rows)

    for idx in tqdm(flagged_df.index, desc="IF AI Row Summaries"):
        row = flagged_df.loc[idx]
        prompt = (
            f"You are a Medicaid fraud expert reviewing claims flagged by an AI anomaly detector (Isolation Forest). "
            f"Here's the claim:\n\n"
            f"Beneficiary: {row['DESYNPUF_ID']}\n"
            f"Claim ID: {row['CLM_ID']}\n"
            f"Type: {row['CLAIM_TYPE']}\n"
            f"Admission Date: {row['CLM_ADMSN_DT']}\n"
            f"Discharge Date: {row['NCH_BENE_DSCHRG_DT']}\n"
            f"Length of Stay: {row['LENGTH_OF_STAY']} days\n"
            f"Total Payment: ${row['CLM_PMT_AMT']:.2f}\n"
            f"Cost per Day: ${row['COST_PER_DAY']:.2f}\n\n"
            f"Why might this claim be considered an anomaly? Please provide clear, succinct reasoning for an analyst or auditor."
        )

        try:
            response = ask_gpt(prompt)
            flagged_df.at[idx, "IF_AI_REASON"] = response.strip() if response else "Empty GPT response."
        except Exception as e:
            flagged_df.at[idx, "IF_AI_REASON"] = "AI summary unavailable."
            if debug:
                print(f"Error for CLM_ID {row['CLM_ID']}: {e}")

    return flagged_df[["CLM_ID", "IF_AI_REASON"]]
