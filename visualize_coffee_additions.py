import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1. Input & setup ----
if len(sys.argv) < 2:
    print("Usage: python visualize_coffee_additions.py <path_to_csv> [--outdir <output_folder>]")
    sys.exit(1)

csv_path = sys.argv[1]
outdir = "coffee_report"
if "--outdir" in sys.argv:
    outdir = sys.argv[sys.argv.index("--outdir") + 1]

os.makedirs(outdir, exist_ok=True)

print(f"📥 Reading data from: {csv_path}")
print(f"📤 Outputs will be saved to: {outdir}")

# ---- 2. Load CSV ----
df = pd.read_csv(csv_path)

# ---- 3. Standardize ingredient flags ----
def normalize_item(s):
    s = s.strip().lower()
    if s in {"no - just black", "black", "just black"}:
        return "Black"
    if "milk" in s or "creamer" in s or "half & half" in s or "half and half" in s or "dairy alternative" in s:
        return "Milk/Creamer"
    if "sugar" in s or "sweetener" in s:
        return "Sugar/Sweetener"
    if "flavor syrup" in s or "syrup" in s:
        return "Flavor Syrup"
    if s == "other" or "cinnamon" in s:
        return "Other"
    if "black" in s and "no" in s:
        return "Black"
    return "Other"

rows = []
for combo, cnt in zip(df.iloc[:, 0], df["count"]):
    parts = [p.strip() for p in str(combo).split(",")]
    flags = set(normalize_item(p) for p in parts if p.strip() != "")
    if not flags:
        flags = {"Other"}
    rows.append({
        "combo_raw": combo,
        "count": int(cnt),
        "Black": int("Black" in flags),
        "Milk/Creamer": int("Milk/Creamer" in flags),
        "Sugar/Sweetener": int("Sugar/Sweetener" in flags),
        "Flavor Syrup": int("Flavor Syrup" in flags),
        "Other": int("Other" in flags)
    })

combos = pd.DataFrame(rows)
combos_sorted = combos.sort_values("count", ascending=False).reset_index(drop=True)

# Save cleaned standardized data
cleaned_csv_path = os.path.join(outdir, "coffee_additions_standardized.csv")
combos_sorted.to_csv(cleaned_csv_path, index=False)
print(f"✅ Cleaned standardized data saved to: {cleaned_csv_path}")

# ---- 4. Build stacked bar chart ----
topN = 15
top = combos_sorted.head(topN).copy()

layers = ["Black", "Milk/Creamer", "Sugar/Sweetener", "Flavor Syrup", "Other"]
colors = {
    "Black": "#4b2e1a",
    "Milk/Creamer": "#d9c3a5",
    "Sugar/Sweetener": "#f0e6d6",
    "Flavor Syrup": "#b36b00",
    "Other": "#9aa0a6"
}

stack_data = {layer: top["count"] * top[layer] for layer in layers}

plt.figure(figsize=(12, 6))
bottom = None
x = range(len(top))
for layer in layers:
    plt.bar(x, stack_data[layer], bottom=bottom, label=layer, color=colors[layer])
    bottom = (stack_data[layer] if bottom is None else bottom + stack_data[layer])

plt.xticks(range(len(top)), [c.replace("No - just black", "Black") for c in top["combo_raw"]], rotation=60, ha="right")
plt.ylabel("Respondents")
plt.title("How people doctor their coffee: ingredient stacks by combination (Top 15)")
plt.legend(title="Ingredient", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# Save figure
plot_path = os.path.join(outdir, "coffee_additions_stack.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Visualization saved to: {plot_path}")
