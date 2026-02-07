import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ----------------- 基础 STL 子句解析器 -----------------

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
        "start": int(start),
        "end": int(end),
        "predicate": pred_str,
        "relation": rel_str,
        "value": value_str
    }

def format_structured_text(parsed):
    return (f"op {parsed['op']} "
            f"start {parsed['start']} "
            f"end {parsed['end']} "
            f"predicate {parsed['predicate']} "
            f"relation {parsed['relation']} "
            f"value {parsed['value']}")

# ----------------- 支持 and / or 的递归解析器 -----------------

def tokenize_stl(expr):
    expr = expr.replace("(", " ( ").replace(")", " ) ")
    return expr.split()

def parse_tokens(tokens):
    def parse_expr(idx):
        if tokens[idx] == "(":
            idx += 1
            left, idx = parse_expr(idx)

            if idx >= len(tokens):
                raise ValueError("表达式结构不完整")
            op = tokens[idx]
            if op not in ("and", "or"):
                raise ValueError(f"未知运算符: {op}")
            idx += 1

            right, idx = parse_expr(idx)

            if idx >= len(tokens) or tokens[idx] != ")":
                raise ValueError("缺失右括号")
            idx += 1

            return {"type": op, "left": left, "right": right}, idx

        else:
            # 抓取一个原子公式，直到最后一个括号
            token_expr = ""
            while idx < len(tokens):
                token_expr += tokens[idx]
                if tokens[idx].endswith(")"):
                    break
                idx += 1
            idx += 1
            return parse_stl(token_expr), idx

    # ✅ 新增：支持多个表达式连接
    ast, idx = parse_expr(0)
    while idx < len(tokens):
        if idx < len(tokens) and tokens[idx] in ("and", "or"):
            op = tokens[idx]
            idx += 1
            right_expr, idx = parse_expr(idx)
            ast = {"type": op, "left": ast, "right": right_expr}
        else:
            break
    return ast


def parse_full_stl(expr):
    tokens = tokenize_stl(expr)
    return parse_tokens(tokens)

# ----------------- AST → 结构化文本 (Clause + Logic) -----------------

def ast_to_semantic_text(node):
    clause_list = []
    logic_list = []

    def dfs(n):
        if "type" in n:
            left = dfs(n["left"])
            right = dfs(n["right"])
            name = f"({left} {n['type'].upper()} {right})"
            logic_list.append(f"logic: {name}")
            return name
        else:
            cid = f"clause{len(clause_list) + 1}"
            clause_list.append(f"{cid}: {format_structured_text(n)}")
            return cid

    dfs(node)
    return "\n".join(clause_list + logic_list)

# ----------------- 文本编码模型 -----------------

class TextEncoder(nn.Module):
    def __init__(self, pretrained_dir, embed_dim=256, dropout=0.1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, local_files_only=True)
        self.bert = BertModel.from_pretrained(pretrained_dir, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, text_list):
        tokens = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.bert.device) for k, v in tokens.items()}
        out = self.bert(**tokens)
        cls_emb = out.last_hidden_state[:, 0]
        mean_emb = out.last_hidden_state.mean(dim=1)
        max_emb, _ = out.last_hidden_state.max(dim=1)
        concat = torch.cat([cls_emb, mean_emb, max_emb], dim=-1)
        concat = self.dropout(concat)
        return self.proj(concat)

