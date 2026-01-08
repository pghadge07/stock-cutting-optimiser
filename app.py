import streamlit as st
import pandas as pd
from math import floor
import itertools
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import io

# ------------------------------------------------------
#  PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Cutting Stock Optimizer", layout="wide")

st.title("🟩 Cutting Stock Optimizer")
st.caption("Create Demand & Stock list, then run optimization.")

# ------------------------------------------------------
#  INITIALIZE SESSION STATE DATAFRAMES
# ------------------------------------------------------
if "demand_list" not in st.session_state:
    st.session_state["demand_list"] = pd.DataFrame({
        "grade": ["D513"],
        "thickness": [0.6],
        "width": [100],
        "length": [80],
        "qty": [1],
    })

if "stock_list" not in st.session_state:
    st.session_state["stock_list"] = pd.DataFrame({
        "grade": ["D513"],
        "thickness": [0.6],
        "width": [1250],
        "length": [1830],
    })


# ======================================================
#  SIDEBY-SIDE EDITORS
# ======================================================
col1, col2 = st.columns(2)

# ---------------- Demand Editor ----------------
with col1:
    st.subheader("📦 Demand List")

    demand_df = st.data_editor(
        st.session_state["demand_list"][["grade", "thickness", "width", "length", "qty"]],
        num_rows="dynamic",
        use_container_width=True,
        key="editable_demand"
    )
    if st.button("Save Demand"):
        st.session_state["demand_list"] = demand_df
        st.success("Demand list updated!")


# ---------------- Stock Editor ----------------
with col2:
    st.subheader("📦 Stock Options")

    stock_df = st.data_editor(
        st.session_state["stock_list"][["grade", "thickness", "width", "length"]],
        num_rows="dynamic",
        use_container_width=True,
        key="editable_stock"
    )
    if st.button("Save Stock"):
        st.session_state["stock_list"] = stock_df
        st.success("Stock list updated!")


st.divider()

# Rotation toggle
allow_rotation = st.checkbox("Allow rotation (90°) when generating patterns", value=False)

# Optimization strategy selector
optimization_strategy = st.selectbox(
    "Optimization Strategy:",
    options=["minimize_waste", "minimize_sheets"],
    format_func=lambda x: "🗑️ Minimize Waste" if x == "minimize_waste" else "📄 Minimize Sheets"
)

# ======================================================
#  CUTTING STOCK SOLVER FUNCTION (demand-based only)
# ======================================================
def solve_cutting_stock(demand, stock_options, allow_rotation=False, optimization_strategy="minimize_waste"):

    # FORCE CLEAN DATA TYPES (fixes float→int crash)
    # ---------------- CLEAN DEMAND DATA ----------------
    required_cols = ["width", "length", "qty"]

    # Drop rows with missing required values
    demand = demand.dropna(subset=required_cols).reset_index(drop=True)

    # Convert safely
    demand["width"] = demand["width"].astype(int)
    demand["length"] = demand["length"].astype(int)
    demand["qty"] = demand["qty"].astype(int)
    demand["thickness"] = demand["thickness"].astype(float)
    demand["grade"] = demand["grade"].astype(str)


    # ---------------- CLEAN STOCK DATA ----------------
    stock_required = ["width", "length"]

    stock_options = stock_options.dropna(subset=stock_required).reset_index(drop=True)

    stock_options["width"] = stock_options["width"].astype(int)
    stock_options["length"] = stock_options["length"].astype(int)
    stock_options["thickness"] = stock_options["thickness"].astype(float)
    stock_options["grade"] = stock_options["grade"].astype(str)


    # Convert demand to internal format
    jobs = []
    for idx, row in demand.iterrows():
        jobs.append({
            "idx": idx,
            "grade": row.grade,
            "thick": float(row.thickness),
            "width": int(row.width),
            "length": int(row.length),
            "qty": int(row.qty),
            "name": f"Job{idx+1}"     # auto assign
        })


    # Convert stock to internal format (preserve row index)
    stocks = []
    for idx, row in stock_options.iterrows():
        stocks.append({
            "stock_idx": idx,
            "name": f"Stock{idx+1}",     # auto assigned human-friendly name
            "grade": row.grade,
            "thick": float(row.thickness),
            "width": int(row.width),
            "length": int(row.length)
        })



    # ---------------- Compatibility check ----------------
    def compatible(s, j):
        return s["grade"] == j["grade"] and abs(s["thick"] - j["thick"]) < 1e-9

    # --------------- PATTERN GENERATION ------------------
    all_patterns = []

    for s in stocks:
        # Build list of compatible jobs (with orientation info if rotation allowed)
        compat_jobs = []
        for j in jobs:
            if not compatible(s, j):
                continue

            # Always consider normal orientation if it fits
            if j["width"] <= s["width"] and j["length"] <= s["length"]:
                compat_jobs.append({
                    **j,
                    "rotated": False,
                    "cut_w": j["width"],
                    "cut_l": j["length"]
                })

            # If rotation is allowed, also consider rotated orientation (if it fits)
            if allow_rotation:
                if j["length"] <= s["width"] and j["width"] <= s["length"]:
                    compat_jobs.append({
                        **j,
                        "rotated": True,
                        "cut_w": j["length"],
                        "cut_l": j["width"]
                    })

        if not compat_jobs:
            continue

        # When rotation is allowed, treat (idx, rotated) as distinct keys.
        if allow_rotation:
            per_row_capacity = {}
            max_rows_per_job = {}
            job_keys = []
            for j in compat_jobs:
                key = (j["idx"], j["rotated"])
                per_row_capacity[key] = s["width"] // j["cut_w"]
                max_rows_per_job[key] = s["length"] // j["cut_l"]
                job_keys.append(key)

            ranges = [range(int(max_rows_per_job[k]) + 1) for k in job_keys]

            for rows_combo in itertools.product(*ranges):
                if all(r == 0 for r in rows_combo):
                    continue

                used_length = 0
                produced = {}
                rotation_map = {}

                for i, rows in enumerate(rows_combo):
                    if rows == 0:
                        continue
                    key = job_keys[i]
                    idx_key, rotated_flag = key
                    # find compat_job entry to know cut_l (search)
                    cj = next(x for x in compat_jobs if x["idx"] == idx_key and x["rotated"] == rotated_flag)
                    used_length += rows * cj["cut_l"]
                    produced[idx_key] = produced.get(idx_key, 0) + rows * int(per_row_capacity[key])
                    if rotated_flag:
                        rotation_map[idx_key] = True
                    else:
                        rotation_map.setdefault(idx_key, False)

                if used_length > s["length"]:
                    continue

                stock_area = s["width"] * s["length"]
                used_area = sum(
                    jobs[jid]["width"] * jobs[jid]["length"] * count
                    for jid, count in produced.items()
                )
                waste_area = stock_area - used_area

                row_breakdown = []   # list of {jid, rotated, per_row, rows, cut_w, cut_l}

                for i, rows in enumerate(rows_combo):
                    if rows == 0:
                        continue

                    key = job_keys[i]  # (jid, rotated)
                    jid, rotated_flag = key
                    cj = next(x for x in compat_jobs if x["idx"] == jid and x["rotated"] == rotated_flag)

                    row_breakdown.append({
                        "jid": jid,
                        "rotated": rotated_flag,
                        "per_row": per_row_capacity[key],
                        "rows": rows,
                        "cut_w": cj["cut_w"],
                        "cut_l": cj["cut_l"]
                    })


                all_patterns.append({
                    "stock_idx": s["stock_idx"],
                    "stock_name": s["name"],
                    "stock_width": s["width"],
                    "stock_length": s["length"],
                    "produced": produced,
                    "rotation_map": rotation_map,
                    "used_length": used_length,
                    "waste_area": waste_area,
                    "row_breakdown": row_breakdown
                })


        else:
            # No rotation: simpler pattern gen (your original approach)
            per_row = {j["idx"]: s["width"] // j["width"] for j in compat_jobs}
            max_rows = {j["idx"]: s["length"] // j["length"] for j in compat_jobs}
            job_ids = [j["idx"] for j in compat_jobs]
            ranges = [range(int(max_rows[j]) + 1) for j in job_ids]

            for combo in itertools.product(*ranges):
                if all(x == 0 for x in combo):
                    continue

                used_length = 0
                produced = {}
                rotation_map = {}
                row_breakdown = []

                for i, rows in enumerate(combo):
                    if rows == 0:
                        continue

                    j = next(j for j in compat_jobs if j["idx"] == job_ids[i])
                    jid = j["idx"]

                    used_length += rows * j["length"]
                    produced[jid] = rows * int(per_row[jid])
                    rotation_map[jid] = False

                    row_breakdown.append({
                        "jid": jid,
                        "rotated": False,
                        "per_row": per_row[jid],
                        "rows": rows,
                        "cut_w": j["width"],
                        "cut_l": j["length"]
                    })

                if used_length > s["length"]:
                    continue

                stock_area = s["width"] * s["length"]
                used_area = sum(
                    jobs[jid]["width"] * jobs[jid]["length"] * cnt
                    for jid, cnt in produced.items()
                )
                waste_area = stock_area - used_area

                all_patterns.append({
                    "stock_idx": s["stock_idx"],
                    "stock_name": s["name"],
                    "stock_width": s["width"],
                    "stock_length": s["length"],
                    "produced": produced,
                    "rotation_map": rotation_map,
                    "row_breakdown": row_breakdown,
                    "waste_area": waste_area,
                })

    # ---------------- ILP MODEL ----------------
    prob = pulp.LpProblem("CuttingStock_DemandBased", pulp.LpMinimize)

    pattern_vars = []
    for i, p in enumerate(all_patterns):
        var = pulp.LpVariable(f"Pattern_{i}_{p['stock_idx']}", lowBound=0, cat="Integer")
        pattern_vars.append((p, var))

    # Objective
    if optimization_strategy == "minimize_waste":
        prob += pulp.lpSum(var * p["waste_area"] for p, var in pattern_vars)
    else:  # minimize_sheets
        prob += pulp.lpSum(var for p, var in pattern_vars)

    # Constraints - meet or exceed demand
    for j in jobs:
        prob += (
            pulp.lpSum(var * p["produced"].get(j["idx"], 0)
                       for p, var in pattern_vars) >= j["qty"]
        )

    prob.solve()
    status = pulp.LpStatus[prob.status]

    return status, pattern_vars, jobs, all_patterns

def transpose_wl(df):
    df2 = df.copy()
    df2["width"], df2["length"] = df2["length"], df2["width"]
    return df2


# ======================================================
#  RUN SOLVER BUTTON
# ======================================================
st.subheader("🚀 Run Cutting Stock Optimization")

if st.button("Run Optimizer"):
    with st.spinner("Solving..."):

        # ---------------- ROW-BASED SOLUTION (current behavior) ----------------
        status_row, patterns_row, jobs_row, _ = solve_cutting_stock(
            st.session_state["demand_list"],
            st.session_state["stock_list"],
            allow_rotation=allow_rotation,
            optimization_strategy=optimization_strategy
        )

        # ---------------- COLUMN-BASED SOLUTION (transpose geometry) ----------------
        demand_col = transpose_wl(st.session_state["demand_list"])
        stock_col  = transpose_wl(st.session_state["stock_list"])

        status_col, patterns_col, jobs_col, _ = solve_cutting_stock(
            demand_col,
            stock_col,
            allow_rotation=allow_rotation,
            optimization_strategy=optimization_strategy
        )

        def total_jobs(patterns):
            total = 0
            for p, v in patterns:
                count = int(pulp.value(v))
                if count > 0:
                    total += sum(p["produced"].values()) * count
            return total


        jobs_row_total = total_jobs(patterns_row)
        jobs_col_total = total_jobs(patterns_col)

        # ================= SELECT BEST GEOMETRY =================
        def solution_metrics(patterns):
            sheets = 0
            waste  = 0
            produced = 0
            for p, v in patterns:
                c = int(pulp.value(v))
                if c > 0:
                    sheets += c
                    waste  += p["waste_area"] * c
                    produced += sum(p["produced"].values()) * c
            return sheets, waste, produced


        row_sheets, row_waste, row_prod = solution_metrics(patterns_row)
        col_sheets, col_waste, col_prod = solution_metrics(patterns_col)

        # 1️⃣ PRIMARY: fewer sheets
        if col_sheets < row_sheets:
            selected_mode = "COLUMN-WISE"
            status   = status_col
            patterns = patterns_col
            jobs     = jobs_col
            geometry_note = "📐 Column-wise cutting selected (fewer sheets used)"

        # 2️⃣ SECONDARY: same sheets → less waste
        elif col_sheets == row_sheets and col_waste < row_waste:
            selected_mode = "COLUMN-WISE"
            status   = status_col
            patterns = patterns_col
            jobs     = jobs_col
            geometry_note = "📐 Column-wise cutting selected (less waste)"

        # 3️⃣ FALLBACK: row-wise
        else:
            selected_mode = "ROW-WISE"
            status   = status_row
            patterns = patterns_row
            jobs     = jobs_row
            geometry_note = "📐 Row-wise cutting selected (tie or better)"



        st.success(f"Solver Status: {status}")

        st.info(geometry_note)


        if status not in ["Optimal", "Feasible"]:
            st.error("No feasible solution found.")
        else:
            # ---------------- Friendly, verified output ----------------
            st.header("📊 Optimization Result (Detailed)")

            # Map job idx -> job name for clear labels
            job_map = {j["idx"]: j["name"] for j in jobs}

            # Show stock list mapping from the session (if available)
            if "stock_list" in st.session_state:
                st.subheader("🗂 Stock index → Stock details")

                stock_map_df = st.session_state["stock_list"].reset_index().rename(columns={"index": "stock_index"})

                # Drop any 'name' column if exists
                if "name" in stock_map_df.columns:
                    stock_map_df = stock_map_df.drop(columns=["name"])

                # Reorder columns by required sequence if present
                desired_order = ["grade", "thickness", "width", "length", "qty"]
                cols_present = [c for c in desired_order if c in stock_map_df.columns]
                other_cols = [c for c in stock_map_df.columns if c not in cols_present and c != "stock_index"]

                stock_map_df = stock_map_df[["stock_index"] + cols_present + other_cols]

                # Show final cleaned dataframe
                st.dataframe(stock_map_df, use_container_width=True)


            # ---------------- OUTPUT in old format ----------------
            total_sheets = 0
            per_stock = {}          # aggregate by stock name
            produced_totals = {j['name']: 0 for j in jobs}

            st.markdown("### === OPTIMAL CUTTING PLAN (Demand-Based) ===")
                
            for p, var in patterns:
                count = int(pulp.value(var))
                if count <= 0:
                    continue

                # label from exact row index
                stock_row_idx = p.get("stock_idx", None)
                if "stock_list" in st.session_state and stock_row_idx is not None:
                    try:
                        stock_label = st.session_state["stock_list"].reset_index().iloc[stock_row_idx]["name"]
                    except Exception:
                        stock_label = p.get("stock_name", f"Stock {stock_row_idx}")
                else:
                    stock_label = p.get("stock_name", f"Stock {stock_row_idx}")

                total_sheets += count
                per_stock[stock_label] = per_stock.get(stock_label, 0) + count

                with st.expander(f"⚙️ Use {count} sheet(s) of Stock {stock_label} ({p['stock_width']}×{p['stock_length']} mm):"):
                    for jid, qty in p["produced"].items():
                        jobname = job_map.get(jid, f"Job {jid}")
                        rotated = p.get("rotation_map", {}).get(jid, False)
                        rotation_label = " (rotated 90°)" if rotated else ""
                        st.markdown(f"- {qty} pcs of **{jobname}**{rotation_label}")
                        produced_totals[jobname] += qty * count

                    st.write(f"   Waste area per sheet: {p['waste_area']:,} mm²")

            st.write(f"\n**Total sheets used:** {total_sheets}")
            st.write(f"**Sheets per stock type:** {per_stock}")

            # ---------------- PRODUCTION SUMMARY ----------------
            st.markdown("\n### === PRODUCTION SUMMARY ===")
            
            for j in jobs:
                demand_qty = j['qty']
                produced_qty = produced_totals[j['name']]
                if produced_qty >= demand_qty:
                    st.write(f"- {j['name']}: Demand={demand_qty}  →  Produced={produced_qty} ✓")
                else:
                    st.write(f"- {j['name']}: Demand={demand_qty}  →  Produced={produced_qty} ⚠️ SHORTFALL")

            # ---------------- WASTE STATS
            total_waste_area = sum(p['waste_area'] * int(pulp.value(v)) for p, v in patterns)
            total_stock_area = sum(p['stock_width'] * p['stock_length'] * int(pulp.value(v)) for p, v in patterns)
            waste_percent = (100 * total_waste_area / total_stock_area) if total_stock_area else 0

            st.write(f"\n**Total waste area:** {total_waste_area:,} mm²")
            st.write(f"**Total waste percentage:** {waste_percent:.2f}%")

            # =========================================
            # VISUALIZATION: Realistic Cutting Layouts on Stock Sheets
            # =========================================
            st.subheader("🖼 Cutting Layout Visualizations")

            used_patterns = [(p, int(pulp.value(v))) for p, v in patterns if int(pulp.value(v)) > 0]

            if not used_patterns:
                st.info("No used patterns to visualize.")
            else:
                all_figures = []
                for p, count in used_patterns:

                    s_width  = p['stock_width']
                    s_length = p['stock_length']

                    # stock name lookup
                    stock_row_idx = p.get("stock_idx", None)
                    if "stock_list" in st.session_state and stock_row_idx is not None:
                        try:
                            stock_name = st.session_state["stock_list"].reset_index().iloc[stock_row_idx]["name"]
                        except:
                            stock_name = f"Stock {stock_row_idx}"
                    else:
                        stock_name = f"Stock {stock_row_idx}"

                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.set_xlim(0, s_width)
                    ax.set_ylim(0, s_length)
                    ax.set_title(f"Cutting Layout – {stock_name} ({s_width}×{s_length} mm) – Use {count} sheet(s)", fontsize=12)
                    ax.set_xlabel("Width (mm)")
                    ax.set_ylabel("Length (mm)")
                    ax.invert_yaxis()

                    # Draw outer sheet
                    ax.add_patch(patches.Rectangle((0, 0), s_width, s_length,
                                                fill=False, edgecolor='black', linewidth=2))

                    color_map = plt.cm.get_cmap('tab20', len(jobs))
                    y_cursor = 0

                    # For legend collection
                    legend_items = {}

                    # Collect placed rectangles for exact waste geometry
                    placed_rects = []

                    pattern_jobs = [(jid, qty) for jid, qty in p["produced"].items()]
                    pattern_jobs.sort(key=lambda x: jobs[x[0]]['length'])

                    for jid, qty in pattern_jobs:

                        job = next(j for j in jobs if j['idx'] == jid)

                        # extract all blocks for this job
                        blocks = [rb for rb in p.get("row_breakdown", []) if rb["jid"] == jid]

                        color = color_map(jid % 20)

                        for block in blocks:
                            cut_w = block["cut_w"]
                            cut_l = block["cut_l"]
                            per_row = block["per_row"]
                            total_rows = block["rows"]
                            rotated = block["rotated"]

                            # Save legend (once per job)
                            if rotated:
                                legend_items[job['name']] = f"{job['length']}×{job['width']} mm (R)"
                            else:
                                legend_items[job['name']] = f"{job['width']}×{job['length']} mm"

                            # Draw all rows of this block and record coords
                            for r in range(total_rows):
                                for c in range(per_row):
                                    x = c * cut_w
                                    y = y_cursor + r * cut_l

                                    rect = patches.Rectangle(
                                        (x, y), cut_w, cut_l,
                                        linewidth=1, edgecolor='black',
                                        facecolor=color, alpha=0.65
                                    )
                                    ax.add_patch(rect)

                                    # record the placed rectangle
                                    placed_rects.append({
                                        "x": float(x),
                                        "y": float(y),
                                        "w": float(cut_w),
                                        "h": float(cut_l),
                                        "jid": jid,
                                        "name": job['name'],
                                        "rotated": rotated
                                    })

                                    # Show job name AND dimensions inside rectangles
                                    ax.text(x + cut_w/2, y + cut_l/2, 
                                           f"{job['name']}\n{int(cut_w)}×{int(cut_l)}",
                                           ha='center', va='center', fontsize=7, weight='bold')

                            # move Y cursor after this block's rows
                            y_cursor += total_rows * cut_l

                    # ----------------------------
                    # Compute & draw exact waste areas (improved logic)
                    # ----------------------------
                    waste_legend = []
                    
                    if placed_rects:
                        # Group rectangles by their Y position to identify rows
                        rows_data = {}
                        for r in placed_rects:
                            y_key = r["y"]
                            if y_key not in rows_data:
                                rows_data[y_key] = {
                                    "rects": [],
                                    "max_x": 0,
                                    "height": r["h"]
                                }
                            rows_data[y_key]["rects"].append(r)
                            rows_data[y_key]["max_x"] = max(rows_data[y_key]["max_x"], r["x"] + r["w"])
                        
                        # Draw waste for each row individually
                        for y_pos in sorted(rows_data.keys()):
                            row = rows_data[y_pos]
                            max_x = row["max_x"]
                            row_height = row["height"]
                            
                            # Right-side waste for this specific row
                            if max_x < s_width:
                                waste_w = s_width - max_x
                                waste_h = row_height
                                
                                if waste_w > 0.5 and waste_h > 0.5:
                                    ax.add_patch(
                                        patches.Rectangle(
                                            (max_x, y_pos),
                                            waste_w,
                                            waste_h,
                                            facecolor='lightgray',
                                            alpha=0.4,
                                            edgecolor='red',
                                            linewidth=1.5,
                                            linestyle='--'
                                        )
                                    )
                                    
                                    waste_area = int(waste_w * waste_h)
                                    waste_legend.append({
                                        "name": f"Waste {len(waste_legend) + 1}",
                                        "w": int(waste_w),
                                        "h": int(waste_h),
                                        "area": waste_area
                                    })
                                    
                                    # Add dimension text
                                    ax.text(
                                        max_x + waste_w/2, 
                                        y_pos + waste_h/2,
                                        f"{int(waste_w)}×{int(waste_h)}",
                                        ha='center', va='center', 
                                        fontsize=7, color='red', 
                                        weight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                                    )
                        
                        # Find overall max Y used
                        if rows_data:
                            max_y_used = max(y + rows_data[y]["height"] for y in rows_data.keys())
                        else:
                            max_y_used = 0
                        
                        # Bottom waste (horizontal strip) - full width
                        if max_y_used < s_length:
                            waste_w_bottom = s_width
                            waste_h_bottom = s_length - max_y_used
                            
                            if waste_w_bottom > 0.5 and waste_h_bottom > 0.5:
                                ax.add_patch(
                                    patches.Rectangle(
                                        (0, max_y_used),
                                        waste_w_bottom,
                                        waste_h_bottom,
                                        facecolor='lightgray',
                                        alpha=0.4,
                                        edgecolor='red',
                                        linewidth=2,
                                        linestyle='--'
                                    )
                                )
                                
                                waste_area_bottom = int(waste_w_bottom * waste_h_bottom)
                                waste_legend.append({
                                    "name": f"Waste {len(waste_legend) + 1}",
                                    "w": int(waste_w_bottom),
                                    "h": int(waste_h_bottom),
                                    "area": waste_area_bottom
                                })
                                
                                # Add dimension text
                                ax.text(
                                    waste_w_bottom/2, 
                                    max_y_used + waste_h_bottom/2,
                                    f"{int(waste_w_bottom)}×{int(waste_h_bottom)}",
                                    ha='center', va='center', 
                                    fontsize=9, color='red', 
                                    weight='bold',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                                )
                    
                    # Fallback if no rectangles were placed
                    if not waste_legend:
                        # Entire sheet is waste
                        waste_w = s_width
                        waste_h = s_length
                        waste_area_total = int(waste_w * waste_h)
                        
                        ax.add_patch(
                            patches.Rectangle(
                                (0, 0),
                                waste_w,
                                waste_h,
                                facecolor='lightgray',
                                alpha=0.4,
                                edgecolor='red',
                                linewidth=2,
                                linestyle='--'
                            )
                        )
                        
                        waste_legend.append({
                            "name": f"Waste {len(waste_legend) + 1}",
                            "w": int(waste_w),
                            "h": int(waste_h),
                            "area": waste_area_total
                        })
                        

                        ax.text(
                            waste_w/2, waste_h/2,
                            f"{int(waste_w)}×{int(waste_h)}",
                            ha='center', va='center', 
                            fontsize=10, color='red', 
                            weight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                        )

                    # ---------------------------------------------------
                    # Build LEGEND inside the plot
                    # ---------------------------------------------------
                    handles = []
                    labels = []
                    for name, dims in legend_items.items():
                        # safe guard: find a job idx for color mapping
                        try:
                            job_idx_for_name = next(j['idx'] for j in jobs if j['name'] == name)
                        except StopIteration:
                            job_idx_for_name = 0
                        patch = patches.Patch(
                            facecolor=color_map(job_idx_for_name % 20),
                            edgecolor='black',
                            alpha=0.65,
                            label=f"{name} – {dims}"
                        )
                        handles.append(patch)
                        labels.append(f"{name} – {dims}")

                    # ---------------- ADD WASTE TO LEGEND ----------------
                    for w in waste_legend:
                        patch = patches.Patch(
                            facecolor='lightgray',
                            edgecolor='red',
                            linestyle='--',
                            linewidth=2,
                            alpha=0.4,
                            label=f"{w['name']} – {w['w']}×{w['h']} mm ({w['area']:,} mm²)"
                        )
                        handles.append(patch)
                        labels.append(
                            f"{w['name']} – {w['w']}×{w['h']} mm ({w['area']:,} mm²)"
                        )

                    # Shrink plot to make space for legend on the right
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

                    ax.legend(
                        handles, labels,
                        loc='center left',
                        bbox_to_anchor=(1, 0.5),
                        fontsize=8,
                        frameon=True,
                        borderpad=0.8,
                        labelspacing=0.5,
                        framealpha=0.95
                    )

                    # Summary waste annotation
                    # ax.text(
                    #     s_width/2, -20,
                    #     f"Total Waste = {p['waste_area']:,} mm²",
                    #     fontsize=10, color='red', ha='center', weight='bold'
                    # )

                    st.pyplot(fig, use_container_width=True)
                    
                    # Store figure for PDF export
                    all_figures.append(fig)
                
                # =========================================
                # PDF EXPORT FUNCTIONALITY
                # =========================================
                if all_figures:
                    st.divider()
                    st.subheader("📥 Export Visualizations")
                    
                    # Create PDF in memory
                    pdf_buffer = io.BytesIO()
                    
                    with PdfPages(pdf_buffer) as pdf:
                        for fig in all_figures:
                            pdf.savefig(fig, bbox_inches='tight')
                    
                    pdf_buffer.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download All Visualizations as PDF",
                        data=pdf_buffer,
                        file_name="cutting_layouts.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )

                    
