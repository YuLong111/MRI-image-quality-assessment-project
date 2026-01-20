# Labels directory

Put your label tables here (CSV/XLSX). This repo includes a copy of `sampled_file_list.xlsx` for demonstration.

Expected columns (from your file):
- `case_id`
- `b_value`
- `score`

Use `scripts/00_make_datalist.py` to convert this to a MONAI datalist JSON.
