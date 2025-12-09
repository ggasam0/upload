from openpyxl import load_workbook
import time
from openpyxl.utils.cell import (
    coordinate_from_string,
    column_index_from_string,
    get_column_letter,
)

# ----------------------------------------------------------------------
# 读取有值或有颜色的单元格
# ----------------------------------------------------------------------
def read_non_empty_or_colored_cells(file_path, sheet_name):
    wb = load_workbook(file_path, data_only=True)
    ws = wb[sheet_name]
    result = []

    for row in ws.iter_rows():
        for cell in row:
            value = cell.value
            fill = cell.fill
            bg_color = None

            if fill and fill.fgColor and fill.fgColor.type == "rgb":
                bg_color = fill.fgColor.rgb

            if value is not None or (bg_color and bg_color != "00000000"):
                result.append((cell.coordinate, value, bg_color))

    return result


# ----------------------------------------------------------------------
# 判断 cell 是否在区域内
# ----------------------------------------------------------------------
def cell_in_range(cell, start, end):
    col_c, row_c = coordinate_from_string(cell)
    col_c = column_index_from_string(col_c)

    col_s, row_s = coordinate_from_string(start)
    col_e, row_e = coordinate_from_string(end)
    col_s, col_e = column_index_from_string(col_s), column_index_from_string(col_e)

    col_min, col_max = sorted([col_s, col_e])
    row_min, row_max = sorted([row_s, row_e])

    return (col_min <= col_c <= col_max) and (row_min <= row_c <= row_max)


# ----------------------------------------------------------------------
# 返回区域外一圈（比区域大一圈）
# ----------------------------------------------------------------------
def get_outer_border(start, end):
    col_s, row_s = coordinate_from_string(start)
    col_e, row_e = coordinate_from_string(end)

    col_s, col_e = column_index_from_string(col_s), column_index_from_string(col_e)
    col_min, col_max = sorted([col_s, col_e])
    row_min, row_max = sorted([row_s, row_e])

    # ---- 防止 col_min 或 row_min 减到 0（Excel 不允许） ----
    col_min = max(1, col_min - 1)
    col_max = col_max + 1

    row_min = max(1, row_min - 1)
    row_max = row_max + 1

    result = []

    # 顶部
    for c in range(col_min, col_max + 1):
        result.append(f"{get_column_letter(c)}{row_min}")

    # 底部
    if row_max != row_min:
        for c in range(col_min, col_max + 1):
            result.append(f"{get_column_letter(c)}{row_max}")

    # 左右
    for r in range(row_min + 1, row_max):
        result.append(f"{get_column_letter(col_min)}{r}")
        result.append(f"{get_column_letter(col_max)}{r}")

    return result



# ----------------------------------------------------------------------
# 在 filled_cells 中找到外圈的第一个匹配
# ----------------------------------------------------------------------
def find_first_match(cells_to_check, filled_cells):
    filled_set = {c for c, v, color in filled_cells}
    for cell in cells_to_check:
        if cell in filled_set:
            return cell
    return None


# ----------------------------------------------------------------------
# 8方向扩张区域
# ----------------------------------------------------------------------
def adjust_region_by_hit_8dir(hit_cell, start, end):
    col_s, row_s = coordinate_from_string(start)
    col_e, row_e = coordinate_from_string(end)
    col_s, col_e = column_index_from_string(col_s), column_index_from_string(col_e)

    col_min, col_max = sorted([col_s, col_e])
    row_min, row_max = sorted([row_s, row_e])

    col_h, row_h = coordinate_from_string(hit_cell)
    col_h = column_index_from_string(col_h)

    new_col_min, new_col_max = col_min, col_max
    new_row_min, new_row_max = row_min, row_max

    # 上/下
    if row_h == row_min - 1:
        new_row_min = row_min - 1
    elif row_h == row_max + 1:
        new_row_max = row_max + 1

    # 左/右
    if col_h == col_min - 1:
        new_col_min = col_min - 1
    elif col_h == col_max + 1:
        new_col_max = col_max + 1

    new_start = f"{get_column_letter(new_col_min)}{new_row_min}"
    new_end   = f"{get_column_letter(new_col_max)}{new_row_max}"

    return new_start, new_end


# ----------------------------------------------------------------------
# ⭐⭐ 主控函数：自动找出所有区域 ⭐⭐
# ----------------------------------------------------------------------
def detect_regions(file_path, sheet_name):
    filled_cells = read_non_empty_or_colored_cells(file_path, sheet_name)

    # filled_cells: list of (coord, value, color)
    # remaining: just coords
    remaining = set(coord for coord, v, c in filled_cells)

    # 为快速查找，将 filled_cells 转为 dict：coord -> (value, color)
    filled_map = {coord: (v, c) for coord, v, c in filled_cells}

    regions = []

    while remaining:
        # 取一个未处理的单元格作为初始区域
        cell = next(iter(remaining))
        start = end = cell

        print("\n==============================")
        print(f"Starting new region from seed: {cell}")
        print("==============================")

        # ============================
        # 区域扩张主循环
        # ============================
        while True:
            outer = get_outer_border(start, end)
            hit = None

            # 找到外圈第一个被填充的单元格
            for oc in outer:
                if oc in filled_map:
                    hit = oc
                    break

            if hit is None:
                print(f"Region stabilized at: {start} → {end}")
                break

            print(f"Hit {hit}, expanding region {start} → {end}")

            # 重要：防止死循环，命中过的外圈格子需要从 remaining / filled_map 中移除
            remaining.discard(hit)
            filled_map.pop(hit, None)

            # 执行扩张
            new_start, new_end = adjust_region_by_hit_8dir(hit, start, end)
            print(f"Expanded to: {new_start} → {new_end}")

            start, end = new_start, new_end

        # ============================
        # 区域确定：移除区域内部所有单元格
        # ============================
        to_remove = [c for c in remaining if cell_in_range(c, start, end)]
        for c in to_remove:
            remaining.remove(c)
            filled_map.pop(c, None)

        regions.append((start, end))
        print(f"Final region added: {start} → {end}")
        print(f"Remaining cells: {len(remaining)}")

    return regions



# ----------------------------------------------------------------------
# 示例
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rs = detect_regions("tes.xlsx", "Sheet1")
    print("检测到的所有区域：")
    for r in rs:
        print(r)
