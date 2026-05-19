def markdown_table(df, max_rows=None):
    if df is None or df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy() if max_rows else df.copy()
    columns = list(view.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in view.to_dict("records"):
        values = []
        for column in columns:
            value = record.get(column, "")
            text = "" if value is None else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    if max_rows and len(df) > max_rows:
        rows.append(f"| ... | {len(df) - max_rows} more rows |" + " |" * max(0, len(columns) - 2))
    return "\n".join(rows)
