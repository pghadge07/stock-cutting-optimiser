import streamlit as st
import pandas as pd
from math import floor
import itertools
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# ---------------- Demand Editor ----------------
with col1:
    st.subheader("📦 Demand List")

    demand_df = st.data_editor(
        st.session_state["demand_list"][["grade", "thickness", "width", "length", "qty"]],
        num_rows="dynamic",
        width="stretch",
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
        width="stretch",
        key="editable_stock"
    )
    if st.button("Save Stock"):
        st.session_state["stock_list"] = stock_df
        st.success("Stock list updated!")


st.divider()

# Rotation toggle
allow_rotation = st.checkbox("Allow rotation (90°) when generating patterns", value=False)

# ======================================================
#  CUTTING STOCK SOLVER FUNCTION (now supports rotation toggle)
# ======================================================
def solve_cutting_stock(demand, stock_options, allow_rotation=False):

    # FORCE CLEAN DATA TYPES (fixes float→int crash)
    # ---------------- CLEAN DEMAND DATA ----------------
    required_cols = ["width", "length", "qty"]

    # Drop rows with missing required values
    demand = demand.dropna(subset=required_cols)

    # Convert safely
    demand["width"] = demand["width"].astype(int)
    demand["length"] = demand["length"].astype(int)
    demand["qty"] = demand["qty"].astype(int)
    demand["thickness"] = demand["thickness"].astype(float)
    demand["grade"] = demand["grade"].astype(str)


    # ---------------- CLEAN STOCK DATA ----------------
    stock_required = ["width", "length"]

    stock_options = stock_options.dropna(subset=stock_required)

    stock_options["width"] = stock_options["width"].astype(int)
    stock_options["length"] = stock_options["length"].astype(int)
    stock_options["thickness"] = stock_options["thickness"].astype(float)
    stock_options["grade"] = stock_options["grade"].astype(str)


    # Convert demand to internal format
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
    # Convert stock to internal format
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
                row_breakdown = []   # ✅ ADD THIS

                for i, rows in enumerate(combo):
                    if rows == 0:
                        continue

                    j = next(j for j in compat_jobs if j["idx"] == job_ids[i])
                    jid = j["idx"]

                    used_length += rows * j["length"]
                    produced[jid] = rows * int(per_row[jid])
                    rotation_map[jid] = False

                    # ✅ ADD ROW BREAKDOWN
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
                    "row_breakdown": row_breakdown,   # ✅ ADD THIS
                    "waste_area": waste_area,
                })

    # ---------------- ILP MODEL ----------------
    prob = pulp.LpProblem("CuttingStock_MinWaste_Rot", pulp.LpMinimize)

    pattern_vars = []
    for i, p in enumerate(all_patterns):
        var = pulp.LpVariable(f"Pattern_{i}_{p['stock_idx']}", lowBound=0, cat="Integer")
        pattern_vars.append((p, var))

    # Objective
    prob += pulp.lpSum(var * p["waste_area"] for p, var in pattern_vars)

    # Constraints
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
            allow_rotation=allow_rotation
        )

        # ---------------- COLUMN-BASED SOLUTION (transpose geometry) ----------------
        demand_col = transpose_wl(st.session_state["demand_list"])
        stock_col  = transpose_wl(st.session_state["stock_list"])

        status_col, patterns_col, jobs_col, _ = solve_cutting_stock(
            demand_col,
            stock_col,
            allow_rotation=allow_rotation
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
                st.dataframe(stock_map_df, width="stretch")


            # ---------------- OUTPUT in old format ----------------
            total_sheets = 0
            per_stock = {}          # aggregate by stock name
            produced_totals = {j['name']: 0 for j in jobs}

            st.markdown("### === OPTIMAL CUTTING PLAN (Rotation Allowed, Minimized Waste) ===")
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
                st.write(f"- {j['name']}: Demand={j['qty']}  →  Produced={produced_totals[j['name']]}")

            # ---------------- WASTE STATS
            total_waste_area = sum(p['waste_area'] * int(pulp.value(v)) for p, v in patterns)
            total_stock_area = sum(p['stock_width'] * p['stock_length'] * int(pulp.value(v)) for p, v in patterns)
            waste_percent = (100 * total_waste_area / total_stock_area) if total_stock_area else 0

            st.write(f"\n**Total waste area:** {total_waste_area:,} mm²")
            st.write(f"**Total waste percentage:** {waste_percent:.2f}%")

            # =========================================
            # VISUALIZATION: Realistic Cutting Layouts on Stock Sheets (Grid-placement waste detection)
            # =========================================
            st.subheader("🖼 Cutting Layout Visualizations")

            used_patterns = [(p, int(pulp.value(v))) for p, v in patterns if int(pulp.value(v)) > 0]

            if not used_patterns:
                st.info("No used patterns to visualize.")
            else:
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

                    fig, ax = plt.subplots(figsize=(8, 3.2))

                    ax.set_xlim(0, s_width)
                    ax.set_ylim(0, s_length)
                    ax.set_title(f"Cutting Layout – {stock_name} ({s_width}×{s_length} mm)", fontsize=12)
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

                    # Collect placed rectangles for exact waste geometry (grid layout coordinates)
                    placed_rects = []   # each item: dict(x, y, w, h, jid, name, rotated)

                    pattern_jobs = [(jid, qty) for jid, qty in p["produced"].items()]
                    pattern_jobs.sort(key=lambda x: jobs[x[0]]['length'])

                    for jid, qty in pattern_jobs:

                        job = next(j for j in jobs if j['idx'] == jid)

                        # extract all blocks for this job (blocks created in pattern gen)
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

                                    # record the placed rectangle for later waste calculation
                                    placed_rects.append({
                                        "x": float(x),
                                        "y": float(y),
                                        "w": float(cut_w),
                                        "h": float(cut_l),
                                        "jid": jid,
                                        "name": job['name'],
                                        "rotated": rotated
                                    })

                                    # ONLY NAME (no dimensions inside rectangles)
                                    ax.text(x + cut_w/2, y + cut_l/2, job['name'],
                                            ha='center', va='center', fontsize=7)

                            # move Y cursor after this block's rows
                            y_cursor += total_rows * cut_l

                    # ----------------------------
                    # Compute & draw exact waste polygons (Option A - grid placement)
                    # ----------------------------
                    waste_legend = []   # <-- ADD THIS

                    try:
                        from shapely.geometry import box as shapely_box, Polygon, MultiPolygon
                        from shapely.ops import polygonize, unary_union
                        shapely_available = True
                    except Exception:
                        shapely_available = False

                    if shapely_available and placed_rects:
                        # build shapely polygons for placed rectangles
                        rect_polys = []
                        for r in placed_rects:
                            # shapely box: (minx, miny, maxx, maxy)
                            rect_polys.append(
                                shapely_box(
                                    r["x"],
                                    s_length - (r["y"] + r["h"]),   # FIX: convert top-origin to bottom-origin
                                    r["x"] + r["w"],
                                    s_length - r["y"]
                                )
                            )


                        used_union = unary_union(rect_polys)
                        sheet_poly = shapely_box(0, 0, s_width, s_length)
                        # raw difference
                        raw_waste = sheet_poly.difference(used_union)

                        # fix invalidities and force ring boundaries
                        clean = raw_waste.buffer(0)

                        # extract individual polygons (splits all disconnected regions)
                        waste_poly = list(polygonize(clean))
                        

                        # draw waste polygons (could be Polygon or MultiPolygon)
                        if isinstance(waste_poly, (Polygon, MultiPolygon)):
                            polys = [waste_poly] if isinstance(waste_poly, Polygon) else list(waste_poly.geoms)
                            for poly in waste_poly:
                                # convert shapely polygon → plotting coords
                                x_coords, y_coords_raw = poly.exterior.xy
                                y_coords = [s_length - y for y in y_coords_raw]

                                ax.fill(
                                    x_coords, y_coords,
                                    facecolor='lightgray', alpha=0.4, edgecolor='black'
                                )

                                # bounding box
                                minx, miny, maxx, maxy = poly.bounds
                                wb = maxx - minx
                                hh = maxy - miny
                                area = poly.area

                                waste_legend.append({
                                    "name": f"Waste {len(waste_legend) + 1}",
                                    "w": int(wb),
                                    "h": int(hh),
                                    "area": int(area)
                                })


                                # label position
                                cx, cy = poly.representative_point().x, poly.representative_point().y

                                # ax.text(
                                #     cx, cy,
                                #     f"WASTE\n{int(wb)}×{int(hh)} mm\n{int(area)} mm²",
                                #     ha='center', va='center', fontsize=8, color='black'
                                # )


                        else:
                            # fallback: if shapely returns something unexpected, show a simple bottom waste
                            if y_cursor < s_length:
                                waste_h = s_length - y_cursor
                                waste_w = s_width
                                ax.add_patch(
                                    patches.Rectangle(
                                        (0, y_cursor),
                                        waste_w,
                                        waste_h,
                                        facecolor='orange',
                                        alpha=0.35,
                                        edgecolor='black',
                                        linewidth=1
                                    )
                                )
                                waste_legend.append({
                                    "name": f"Waste {len(waste_legend) + 1}",
                                    "w": int(waste_w),
                                    "h": int(waste_h),
                                    "area": int(waste_w * waste_h)
                                })

                    else:
                        # shapely not available or no placed rects — fallback to simple bottom waste rectangle
                        if y_cursor < s_length:
                            waste_h = s_length - y_cursor
                            waste_w = s_width
                            ax.add_patch(
                                patches.Rectangle(
                                    (0, y_cursor), waste_w, waste_h,
                                    facecolor='lightgray', alpha=0.35,
                                    edgecolor='black', linewidth=1
                                )
                            )
                            waste_legend.append({
                                "name": f"Waste {len(waste_legend) + 1}",
                                "w": int(waste_w),
                                "h": int(waste_h),
                                "area": int(waste_w * waste_h)
                            })

                            # # Center position
                            # cx = waste_w / 2
                            # cy = y_cursor + waste_h / 2
                            # waste_text = f"WASTE\n{int(waste_w)}×{int(waste_h)} mm\n{int(waste_w * waste_h)} mm²"
                            # # ax.text(cx, cy, waste_text, ha='center', va='center', fontsize=9, color='black')

                    # ---------------------------------------------------
                    # Build LEGEND inside the plot (top-right corner)
                    # ---------------------------------------------------
                    handles = []
                    labels = []
                    for name, dims in legend_items.items():
                        # safe guard: find a job idx for color mapping; if not found default 0
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
                            edgecolor='black',
                            alpha=0.35,
                            label=f"{w['name']} – {w['w']}×{w['h']} mm ({w['area']} mm²)"
                        )
                        handles.append(patch)
                        labels.append(
                            f"{w['name']} – {w['w']}×{w['h']} mm ({w['area']} mm²)"
                        )


                    # Shrink plot to make space for legend on the right
                    box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])   # 22% extra space on right

                    ax.legend(
                        handles, labels,
                        loc='upper right',
                        fontsize=8,
                        frameon=True,
                        borderpad=0.4,
                        labelspacing=0.3,
                        framealpha=0.9
                    )

                    # Summary waste annotation (keeps original red text)
                    ax.text(
                        s_width/2, s_length - 8,
                        f"Total Waste = {p['waste_area']} mm²",
                        fontsize=9, color='red', ha='center'
                    )

                    st.pyplot(fig, use_container_width=True)

