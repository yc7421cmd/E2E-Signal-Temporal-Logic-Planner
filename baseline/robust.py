import torch
import re
import json


with open("area1.json") as f:
    region_dict1 = json.load(f)


# ---------- STL 子句解析 ----------
def parse_stl(expr):
    expr = expr.replace(" (", "(").replace("( ", "(").replace(" )", ")").replace(") ", ")")
    m = re.match(r"(\w)\[(\d+),(\d+)\]\((.+)\)", expr)
    if not m:
        raise ValueError(f"无法解析表达式: {expr}")
    op_sym, start, end, condition = m.groups()
    op_map = {"F": "eventually", "G": "always", "U": "until"}
    op_str = op_map.get(op_sym, op_sym)

    m2 = re.match(r"(\w+)\s*(in|not in|[<>=!]+)\s*(\w+)", condition)
    if not m2:
        raise ValueError(f"无法解析条件: {condition}")
    pred_str, rel_str, value_str = m2.groups()

    return {
        "op": op_str,
        "start": start,
        "end": end,
        "predicate": pred_str,
        "relation": rel_str,
        "value": value_str
    }

def tokenize_stl(expr):
    expr = expr.replace("(", " ( ").replace(")", " ) ")
    return expr.split()

def parse_tokens(tokens):
    def parse_expr(idx):
        if tokens[idx] == "(":
            idx += 1
            left, idx = parse_expr(idx)
            op = tokens[idx]
            if op not in ("and", "or"):
                raise ValueError(f"未知运算符: {op}")
            idx += 1
            right, idx = parse_expr(idx)
            if tokens[idx] != ")":
                raise ValueError("缺失右括号")
            idx += 1
            return {"type": op, "left": left, "right": right}, idx
        else:
            token_expr = ""
            while idx < len(tokens):
                token_expr += tokens[idx]
                if tokens[idx].endswith(")"):
                    break
                idx += 1
            idx += 1
            return parse_stl(token_expr), idx

    ast, idx = parse_expr(0)
    while idx < len(tokens):
        if tokens[idx] in ("and", "or"):
            op = tokens[idx]
            idx += 1
            right, idx = parse_expr(idx)
            ast = {"type": op, "left": ast, "right": right}
        else:
            break
    return ast

def parse_full_stl(expr):
    tokens = tokenize_stl(expr)
    return parse_tokens(tokens)

# ---------- 平滑 min/max ----------
k = 10.0  # 可调
def smooth_min(lis):
    lis = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in lis]
    return -torch.logsumexp(torch.stack([-k * v for v in lis]), dim=0) / k

def smooth_max(lis):
    lis = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in lis]
    return torch.logsumexp(torch.stack([k * v for v in lis]), dim=0) / k

# ---------- F/G 鲁棒度 ----------
def get_F(start, end, traj, region_bounds):
    x_min = region_bounds['x_min']
    x_max = region_bounds['x_max']
    y_min = region_bounds['y_min']
    y_max = region_bounds['y_max']
    delta = []
    for i in range(start, end + 1):
        x, y = traj[i, 0], traj[i, 1]
        delta_step = [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y
        ]
        delta.append(smooth_min(delta_step))
    return smooth_max(delta)

def get_G(start, end, traj, region_bounds):
    x_min = region_bounds['x_min']
    x_max = region_bounds['x_max']
    y_min = region_bounds['y_min']
    y_max = region_bounds['y_max']
    delta = []
    for i in range(start, end + 1):
        x, y = traj[i, 0], traj[i, 1]
        delta_step = [
            x - x_min,
            x_max - x,
            y - y_min,
            y_max - y
        ]
        delta.append(smooth_min(delta_step))
    return smooth_min(delta)

# ---------- 递归 AST 鲁棒度评估 ----------
def eval_ast(ast_node, traj):
    if "type" in ast_node:
        left_val = eval_ast(ast_node["left"], traj)
        right_val = eval_ast(ast_node["right"], traj)
        if ast_node["type"] == "and":
            return smooth_min([left_val, right_val])
        elif ast_node["type"] == "or":
            return smooth_max([left_val, right_val])
        else:
            raise ValueError(f"未知逻辑类型: {ast_node['type']}")
    else:
        op = ast_node['op']
        start = int(ast_node["start"])
        end = int(ast_node["end"])
        region = ast_node["value"]
        region_bounds = region_dict1[region]


        if op == "eventually":
            res = get_F(start, end, traj, region_bounds)
        elif op == "always":
            res = get_G(start, end, traj, region_bounds)
        else:
            raise ValueError(f"未知操作符: {op}")

        # # ✅ 加在这里！
        # print(f"[DEBUG] {op.upper()}[{start},{end}] in {region} → robustness: {res.item():.4f}")
        return res


# ---------- 批量鲁棒度 ----------
# 修改 robust_cal:
def robust_cal(stl_list, pred_traj, low=-2, r=0.2):
    total = 0
    batch = len(stl_list)
    for i in range(batch):
        expr = stl_list[i]
        traj = pred_traj[i, :, :]
        try:
            ast = parse_full_stl(expr)
            val = eval_ast(ast, traj)
            # ✅ 不要 min(r, val) 在这里裁剪（否则正值被削掉了）
            total += val
        except Exception as e:
            print(f"[STL解析失败] {expr}, 错误: {e}")
            total += low
    avg = float(total / batch)
    # ✅ 裁剪整个 batch 平均后再 min/max
    return max(min(avg, r), low)

