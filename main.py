import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from data.snowflake_connector import (
    load_inpatient_claims,
    load_outpatient_claims,
    load_beneficiary_data,
    merge_claims_with_beneficiaries,
)

from analytics.analytics_core import (
    engineer_core_features,
    flag_rules_with_progress,
    evaluate_predictions,
    add_isolation_forest_scores,
    summarize_isolation_forest_flags,
)

from llm.openai_client import ask_gpt

# Cap GPT summaries for cost/performance reasons
MAX_GPT_SUMMARIES = 10


def assign_flag_source(row):
    if row["IS_FLAGGED"] == "Y" and row["IF_FLAGGED"] == "Y":
        return "BOTH"
    elif row["IS_FLAGGED"] == "Y":
        return "RULE_ONLY"
    elif row["IF_FLAGGED"] == "Y":
        return "IF_ONLY"
    return "NONE"


def main():
    total_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_results_{timestamp}.txt"

    print("Medicare FWA Analysis - Proof of Concept")

    try:
        with open(log_filename, "w") as log:
            # Fetch data
            inpatient_df = load_inpatient_claims()
            outpatient_df = load_outpatient_claims()
            beneficiary_df = load_beneficiary_data()

            # Merge
            merged_df = merge_claims_with_beneficiaries(inpatient_df, outpatient_df, beneficiary_df)
            log.write(f"Merged claims total: {len(merged_df):,} rows\n")

            # Core Feature Engineering
            start_core = time.time()
            merged_df = engineer_core_features(merged_df)
            core_time = time.time() - start_core
            log.write(f"Core features done in {core_time:.2f} sec\n")

            # Rule-based Flagging
            start_flagging = time.time()
            enriched_df = flag_rules_with_progress(merged_df)
            flagging_time = time.time() - start_flagging
            log.write(f"Rule flagging done in {flagging_time:.2f} sec\n")
            log.write("\nRule Breakdown:\n")
            log.write(f"  > 0-day high pay: {enriched_df['RULE_ZERO_DAYS_HIGH_PAY'].sum()}\n")
            log.write(f"  > Duplicate CLM_ID: {enriched_df['RULE_DUPLICATE_CLAIM_ID'].sum()}\n")
            log.write(f"  > Flagged total: {enriched_df['IS_FLAGGED'].value_counts().to_dict()}\n")

            # Rule-based evaluation
            rule_report_text = evaluate_predictions(enriched_df, print_report=False)
            log.write("\nRule-based Evaluation Metrics:\n")
            log.write(rule_report_text + "\n")

            # Isolation Forest scoring
            enriched_df = add_isolation_forest_scores(enriched_df)
            if_report_text = evaluate_predictions(enriched_df, predicted_column="IF_FLAGGED", print_report=False)
            log.write("\nIsolation Forest Evaluation Metrics:\n")
            log.write(if_report_text + "\n")

            # GPT Summary of IF performance
            log.write("\nGPT Summary of Isolation Forest Evaluation:\n")
            try:
                summary_prompt = (
                    "You are a healthcare fraud analyst. A machine learning model (Isolation Forest) "
                    "was used to detect anomalies in Medicare claims. Please summarize the evaluation "
                    "metrics below in 2-3 plain-language bullet points for non-technical stakeholders.\n\n"
                    f"```\n{if_report_text}\n```"
                )
                gpt_summary = ask_gpt(summary_prompt, max_tokens=250)
                log.write(gpt_summary + "\n")
            except Exception as e:
                log.write(f"⚠️ GPT Summary failed: {e}\n")

            # GPT summaries for IF-flagged claims
            if_ai_summaries = summarize_isolation_forest_flags(enriched_df, debug=True, max_rows=MAX_GPT_SUMMARIES)
            enriched_df = enriched_df.merge(if_ai_summaries, on="CLM_ID", how="left")

            # Final label
            enriched_df["FLAG_SOURCE"] = enriched_df.apply(assign_flag_source, axis=1)

            # Export CSV
            enriched_df.to_csv("flagged_claims_summary.csv", index=False)
            log.write("\nSaved results to 'flagged_claims_summary.csv'\n")
            log.write(f"Processed {len(enriched_df):,} claims.\n")
            log.write(f"Rule-flagged: {enriched_df['IS_FLAGGED'].value_counts().to_dict()}\n")

    except Exception as e:
        print(f"Error during execution: {e}")

    print(f"\n⏱ Total run time: {time.time() - total_start:.2f} seconds.")


if __name__ == "__main__":
    main()
