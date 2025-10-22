#!/usr/bin/env python3

# Quick fix for database field truncation issues

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the remaining p_value_text line
old_line = "p_value_text=comparison.get('p_value', '').split(' - Source:')[0].strip(),"
new_line = "p_value_text=_safe_truncate(comparison.get('p_value', '')),"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Fixed p_value_text truncation")
else:
    print("❌ p_value_text line not found or already fixed")

# Also fix any remaining timepoint_unit and timepoint_type issues
content = content.replace(
    "timepoint_unit=timepoint_unit,",
    "timepoint_unit=_safe_truncate(timepoint_unit),"
)

content = content.replace(
    "timepoint_type=tp_data.get('timepoint_type', outcome_type),",
    "timepoint_type=_safe_truncate(tp_data.get('timepoint_type', outcome_type)),"
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ All database field truncation issues fixed")