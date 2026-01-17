import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION: RESEARCH PAPER STYLE ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def load_benchmark_data():
    json_files = glob.glob("benchmark_results_*.json")
    data_list = []

    for file in json_files:
        try:
            with open(file, "r") as f:
                content = json.load(f)
                entries = []
                
                # --- LOGIC FOR SINGLE MODEL FILES ---
                if "detailed_runs" in content and "summary" in content:
                    raw_name = content["summary"].get("model", "Unknown")
                    
                    # 1. Determine Precision
                    # Check filename first
                    if "4bit" in file or "4-bit" in file: precision = "INT4"
                    elif "8bit" in file or "8-bit" in file: precision = "INT8"
                    elif "fp16" in file: precision = "FP16"
                    else: precision = "Unspecified"
                    
                    # 2. Apply User Overrides
                    if precision == "Unspecified":
                        if "phi-3.5" in raw_name.lower():
                            precision = "INT8"
                        else:
                            precision = "INT4"
                            
                    # 3. Format Name
                    if "phi-3.5" in raw_name.lower():
                        display_name = f"Phi-3.5-Mini 3.8B ({precision})"
                    else:
                        display_name = f"{raw_name} ({precision})"

                    runs = content["detailed_runs"]
                    avg_latency = np.mean([r.get('latency', 0) for r in runs])
                    avg_ttft = np.mean([r.get('ttft', 0) for r in runs])
                    
                    entries.append({
                        "Model": display_name,
                        "Latency (s)": avg_latency,
                        "TTFT (s)": avg_ttft,
                        "Throughput (tok/s)": content["summary"].get("throughput", 0),
                        "BERTScore (F1)": content["summary"].get("bert_score_f1", 0),
                    })
                    
                # --- LOGIC FOR MULTI-MODEL FILES (Tiny Giants) ---
                elif isinstance(content, dict):
                    for name, stats in content.items():
                        # Tiny Giants are explicitly FP16
                        entries.append({
                            "Model": f"{name} (FP16)",
                            "Latency (s)": np.mean([r['latency'] for r in stats['runs']]),
                            "TTFT (s)": np.mean([r['ttft'] for r in stats['runs']]),
                            "Throughput (tok/s)": stats.get("throughput", 0),
                            "BERTScore (F1)": stats.get("bert_score", 0)
                        })

                data_list.extend(entries)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            
    return pd.DataFrame(data_list)

def create_research_graph(df):
    plt.figure(figsize=(10, 6))
    
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='#d3d3d3', zorder=0)
    
    sns.scatterplot(
        data=df,
        x="Throughput (tok/s)",
        y="BERTScore (F1)",
        hue="Model",
        style="Model",
        s=120, 
        palette="viridis",
        zorder=3,
        edgecolor='black',
        linewidth=0.5
    )

    # Annotations
    for i, row in df.iterrows():
        plt.text(
            row["Throughput (tok/s)"] + (df["Throughput (tok/s)"].max() * 0.015),
            row["BERTScore (F1)"] + 0.0005, 
            row["Model"],
            fontsize=8,
            color='#333333',
            verticalalignment='bottom'
        )

    plt.title("Performance Frontier: Throughput vs. Generation Quality", pad=20, weight='bold')
    plt.xlabel("Throughput (Tokens / Second)\n(Higher is Better)", labelpad=10)
    plt.ylabel("BERTScore F1 (Semantic Similarity)\n(Higher is Better)", labelpad=10)

    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True, fancybox=False, edgecolor='#d3d3d3', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("benchmark_graph_research.png")
    print("Graph saved to 'benchmark_graph_research.png'")

def create_data_table_image(df):
    # Removed TPOT from columns
    cols = ["Model", "Latency (s)", "TTFT (s)", "Throughput (tok/s)", "BERTScore (F1)"]
        
    sub_df = df[cols].copy().sort_values("BERTScore (F1)", ascending=False).reset_index(drop=True)
    sub_df = sub_df.round(4)

    fig, ax = plt.subplots(figsize=(16, len(sub_df) * 0.8 + 3))
    ax.axis('off')

    # UPDATED COLUMN WIDTHS
    # Removed TPOT width (0.12) and gave most of it to Model (0.30 -> 0.40)
    widths_map = {
        "Model": 0.40,
        "Latency (s)": 0.13,
        "TTFT (s)": 0.13,
        "Throughput (tok/s)": 0.17,
        "BERTScore (F1)": 0.17
    }
    final_widths = [widths_map[c] for c in sub_df.columns]

    table = ax.table(
        cellText=sub_df.values,
        colLabels=sub_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
        colWidths=final_widths 
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    min_cols = ["Latency (s)", "TTFT (s)"]
    max_cols = ["Throughput (tok/s)", "BERTScore (F1)"]

    # Header Styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_height(0.15)
        else:
            cell.set_height(0.1)
            if j == 0: cell.set_text_props(ha='left')

    # Bolding the Best Values
    for col_idx, col_name in enumerate(sub_df.columns):
        if col_idx == 0: continue
        values = pd.to_numeric(sub_df[col_name], errors='coerce')
        target_val = None
        if col_name in min_cols: target_val = values.min()
        elif col_name in max_cols: target_val = values.max()
            
        if target_val is not None and not pd.isna(target_val):
            for row_idx, val in enumerate(values):
                if abs(val - target_val) < 1e-9: 
                    cell = table[row_idx + 1, col_idx]
                    cell.set_text_props(weight='bold', color='#D21404')
                    cell.set_facecolor('#f9e6e6')

    plt.title("Detailed Model Performance Metrics", weight='bold', pad=20)
    plt.savefig("benchmark_table_export.png", dpi=300, bbox_inches='tight')
    print("Table saved to 'benchmark_table_export.png'")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading data...")
    df = load_benchmark_data()
    
    if not df.empty:
        print(f"Loaded {len(df)} model runs.")
        create_research_graph(df)
        create_data_table_image(df)
    else:
        print("No data found in json files.")