import pandas as pd
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.evaluation.single_table import get_column_plot

def run_final_metrics():
    # 1. LOAD THE NEW DATA
    try:
        # We load the "Expert Seed" as the ground truth
        real_data = pd.read_csv('expert_seed.csv')
        # We load the "V3 Optimized" synthetic data
        synthetic_data = pd.read_csv('triage_v3_90plus.csv')
        print("--- FILES LOADED SUCCESSFULLY ---")
    except FileNotFoundError:
        print("--- ERROR: Could not find 'expert_seed.csv' or 'triage_v3_90plus.csv' ---")
        return

    # 2. SETUP METADATA
    metadata = Metadata.detect_from_dataframe(data=real_data)

    # 3. RUN QUALITY REPORT
    print("Computing Final Metrics... This analyzes how well the AI learned medical logic.")
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    # 4. RUN DIAGNOSTICS
    diagnostic_report = run_diagnostic(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    # 5. PRINT THE RESULTS FOR THE JUDGES
    print("\n" + "="*50)
    print("V3 TRIAGE SYSTEM QUALITY REPORT")
    print("="*50)
    
    overall_score = quality_report.get_score()
    print(f"OVERALL QUALITY SCORE: {overall_score * 100:.2f}%")
    
    print("\nDETAILED BREAKDOWN:")
    shapes_score = quality_report.get_details('Column Shapes')['Score'].mean()
    trends_score = quality_report.get_details('Column Pair Trends')['Score'].mean()
    
    print(f"- Data Distribution (Shapes): {shapes_score * 100:.2f}%")
    print(f"- Medical Logic (Pair Trends): {trends_score * 100:.2f}%")
    print("="*50)

    # 6. VISUAL VALIDATION (Saves an image for your presentation)
    print("\nGenerating visual proof...")
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name='Risk_Level',
        metadata=metadata
    )
    fig.write_image("final_validation_chart.png")
    print("SUCCESS: 'final_validation_chart.png' saved. Put this in your slides!")

if __name__ == "__main__":
    run_final_metrics()