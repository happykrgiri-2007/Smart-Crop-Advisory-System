
# ============================================================
# SMART CROP ADVISORY SYSTEM - INTEGRATION
# Connects Module 1, Module 2 and Module 3
# ============================================================

import os
import runpy
import glob
from pathlib import Path

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1: FIND PROJECT AND MODULE FOLDERS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

def find_module_folder(prefix):
    folders = [
        path for path in PROJECT_DIR.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]

    if not folders:
        raise FileNotFoundError(
            f"Cannot find a folder starting with {prefix}"
        )

    if len(folders) > 1:
        raise RuntimeError(
            f"More than one folder starts with {prefix}: {folders}"
        )

    return folders[0]


module1_dir = find_module_folder("Module-1")
module2_dir = find_module_folder("Module-2")
module3_dir = find_module_folder("Module-3")

print("Module 1 folder:", module1_dir.name)
print("Module 2 folder:", module2_dir.name)
print("Module 3 folder:", module3_dir.name)


# ------------------------------------------------------------
# STEP 2: LOAD EACH MODULE
# ------------------------------------------------------------

def load_python_module(folder, script_pattern):
    scripts = list(folder.glob(script_pattern))

    if len(scripts) != 1:
        raise FileNotFoundError(
            f"Expected one file matching {script_pattern} "
            f"in {folder.name}, found: {[p.name for p in scripts]}"
        )

    script_path = scripts[0]
    old_directory = os.getcwd()

    try:
        # CSV files are loaded relative to each module folder.
        os.chdir(folder)

        # Existing scripts contain plots. Close plots instead
        # of opening chart windows during integration.
        plt.show = lambda *args, **kwargs: plt.close("all")

        namespace = runpy.run_path(str(script_path))
        print(f"\nLoaded: {script_path.name}")
        return namespace

    finally:
        os.chdir(old_directory)


print("\nLoading Module 1...")
module1 = load_python_module(
    module1_dir, "02_*.py"
)

print("\nLoading Module 2...")
module2 = load_python_module(
    module2_dir, "02_*.py"
)

print("\nLoading Module 3...")
module3 = load_python_module(
    module3_dir, "02_*.py"
)


# ------------------------------------------------------------
# STEP 3: GET THE THREE REUSABLE FUNCTIONS
# ------------------------------------------------------------

recommend_crop = module1.get("recommend_crop")
predict_yield = module2.get("predict_yield")
get_profit_estimate = module3.get("get_profit_estimate")

if not all([
    callable(recommend_crop),
    callable(predict_yield),
    callable(get_profit_estimate)
]):
    raise RuntimeError(
        "One or more required functions were not found. "
        "Check the function names in all three modules."
    )

print("\nAll three module functions loaded successfully!")


# ------------------------------------------------------------
# STEP 4: COMBINED ADVISORY FUNCTION
# ------------------------------------------------------------

def smart_crop_advisory(
    N, P, K,
    temperature, humidity, ph, rainfall,
    state, district, season, year, soil_moisture
):
    print("\n" + "=" * 55)
    print("           SMART CROP ADVISORY REPORT")
    print("=" * 55)

    # Module 1: Recommend crop
    try:
        crop = recommend_crop(
            N, P, K, temperature, humidity, ph, rainfall
        )

        if crop is None or not str(crop).strip():
            print("Recommended crop: Not available")
            return

        crop_lower = str(crop).strip().lower()
        print("\n1. RECOMMENDED CROP:", crop_lower.title())

    except Exception as error:
        print("\nCrop recommendation failed:", error)
        return

    # Module 2: Predict yield
    print("\n2. CROP YIELD PREDICTION")

    try:
        yield_result = predict_yield(
            state.strip(),
            district.strip(),
            season.strip(),
            crop_lower.title(),
            year,
            temperature,
            humidity,
            soil_moisture
        )

        if yield_result is None:
            print("Yield: Not enough matching data.")
        else:
            print(f"Predicted yield: {yield_result:.4f}")

    except (ValueError, KeyError, LookupError) as error:
        yield_result = None
        print("Yield unavailable:", error)

    except Exception as error:
        yield_result = None
        print("Yield prediction failed:", error)

    # Module 3: Estimate profit
    print("\n3. CROP PROFITABILITY")

    try:
        profit_result = get_profit_estimate(
            crop_lower.upper(), state
        )

        if profit_result is None:
            print("Profit: No matching crop-state record.")
        elif isinstance(profit_result, dict):
            profit = profit_result.get(
                "Estimated_Profit_per_Hectare"
            )

            if profit is None:
                print("Profit estimate not present in result.")
            else:
                print(f"Estimated profit per hectare: {profit:,.2f}")

            print("Profit record state:", profit_result.get("State"))
            print("Support price:", profit_result.get("Support_Price"))
            print(
                "Historical yield per hectare:",
                profit_result.get("Yield_Quintal_Hectare")
            )
        else:
            print("Profit result:", profit_result)

    except Exception as error:
        profit_result = None
        print("Profit estimate unavailable:", error)

    # Final summary
    print("\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print("Recommended crop:", crop_lower.title())
    print(
        "Yield result:",
        yield_result if yield_result is not None else "Not available"
    )

    if profit_result is None:
        print("Profit result: Not available")
    elif isinstance(profit_result, dict):
        print("Profit result: Historical record found")
    else:
        print("Profit result:", profit_result)

    print("\nMissing data is reported, not guessed.")
    print("=" * 55)


# ------------------------------------------------------------
# STEP 5: SUCCESS-ATTEMPT TEST
# ------------------------------------------------------------

# Use a real Module 2 row for location/season/year inputs.
# The recommended crop may still be missing from another dataset.

module2_df = module2.get("df")

if module2_df is None or module2_df.empty:
    raise RuntimeError("Module 2 dataset is empty or unavailable.")

sample = module2_df.iloc[0]


print("\n\nRUNNING INTEGRATION TEST")

# Use matching crop and state for Module 2 and Module 3
smart_crop_advisory(
    N=90,
    P=42,
    K=43,
    temperature=float(sample["Temperature"]),
    humidity=float(sample["Humidity"]),
    ph=6.5,
    rainfall=200.0,
    state="Bihar",
    district=str(sample["District_Name"]),
    season=str(sample["Season"]),
    year=int(sample["Crop_Year"]),
    soil_moisture=float(sample["Soil_Moisture"])
)



# ------------------------------------------------------------
# STEP 6: LIMITATION TEST FOR MISSING PROFIT RECORD
# ------------------------------------------------------------

print("\n\nLIMITATION TEST: UNKNOWN CROP-STATE")

missing_result = get_profit_estimate(
    "UNKNOWN CROP",
    "Unknown State"
)

if missing_result is None:
    print("PASS: Missing crop-state record handled correctly.")
else:
    print("Check the result:", missing_result)


print("\nINTEGRATION SCRIPT FINISHED.")
# CHECK AVAILABLE CROPS AND STATES

print("\n========== DATASET MATCHING CHECK ==========")

# Module 2: Crop Yield Dataset
df_yield = module2.get("df")

if df_yield is not None:
    print("\nMODULE 2 - AVAILABLE CROPS:")
    print(sorted(df_yield["Crop"].dropna().astype(str).str.strip().unique()))

    print("\nMODULE 2 - AVAILABLE STATES:")
    print(sorted(df_yield["State_Name"].dropna().astype(str).str.strip().unique()))

# Module 3: Profitability Dataset
df_profit = module3.get("df")

if df_profit is not None:
    print("\nMODULE 3 - AVAILABLE CROPS:")
    print(sorted(df_profit["Crop"].dropna().astype(str).str.strip().unique()))

    print("\nMODULE 3 - AVAILABLE STATES:")
    print(sorted(df_profit["State"].dropna().astype(str).str.strip().unique()))

print("\n========== CHECK COMPLETE ==========")
print("\nINTEGRATION SCRIPT FINISHED.")
print("\n========== COMMON CROP CHECK ==========")

df_recommend = module1.get("df")
df_yield = module2.get("df")
df_profit = module3.get("df")

if df_recommend is not None:
    module1_crops = set(
        df_recommend["label"].dropna().astype(str).str.strip().str.lower()
    )
    print("\nModule 1 crops:", sorted(module1_crops))

if df_yield is not None:
    module2_crops = set(
        df_yield["Crop"].dropna().astype(str).str.strip().str.lower()
    )
    print("\nModule 1 + Module 2 common crops:")
    print(sorted(module1_crops & module2_crops))

if df_profit is not None:
    module3_crops = set(
        df_profit["Crop"].dropna().astype(str).str.strip().str.lower()
    )
    print("\nModule 1 + Module 3 common crops:")
    print(sorted(module1_crops & module3_crops))

print("\n========== COMMON CROP CHECK COMPLETE ==========")
print("\n========== MAIZE STATE MATCH CHECK ==========")

df_yield = module2.get("df")
df_profit = module3.get("df")

if df_yield is not None:
    print("\nModule 2 - Maize states:")
    print(sorted(
        df_yield.loc[
            df_yield["Crop"].astype(str).str.strip().str.lower() == "maize",
            "State_Name"
        ].dropna().astype(str).str.strip().unique()
    ))

if df_profit is not None:
    print("\nModule 3 - Maize states:")
    print(sorted(
        df_profit.loc[
            df_profit["Crop"].astype(str).str.strip().str.lower() == "maize",
            "State"
        ].dropna().astype(str).str.strip().unique()
    ))

print("\n========== STATE MATCH CHECK COMPLETE ==========")
print("\n===== MAIZE + BIHAR PROFIT TEST =====")

profit_test = get_profit_estimate("MAIZE", "Bihar")

if profit_test is not None:
    print("Profit record found:")
    print(profit_test)
else:
    print("No matching Maize-Bihar profit record.")