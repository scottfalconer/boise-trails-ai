#!/usr/bin/env python3
import json, sys, math
from pathlib import Path
from collections import defaultdict
BASE = Path(__file__).resolve().parent
cert = json.load(open(BASE/'boise-positive-home-schedule-certificate-v2.0.json'))
universe = json.load(open(BASE/'positive-run-loop-universe-v2.0.json'))
field_days = json.load(open(BASE/'positive-field-day-candidates-v2.0.json'))
schedule = json.load(open(BASE/'positive-selected-home-schedule-v2.0.json'))
optimizer = json.load(open(BASE/'positive-schedule-optimizer-certificate-v2.0.json'))
errors=[]
B = cert['daily_bounds']
loops={x['loop_id']:x for x in universe['run_loops']}
if cert.get('verdict') != 'positive_certificate_issued': errors.append('certificate verdict is not positive')
if not cert.get('achieved'): errors.append('certificate achieved flag is false')
for loop in loops.values():
    if loop['on_foot_miles'] > B['max_daily_on_foot_miles'] + 1e-9:
        errors.append(f"loop {loop['loop_id']} exceeds daily on-foot bound")
cover=defaultdict(int)
for fd in schedule['selected_field_days']:
    if fd['on_foot_miles'] > B['max_daily_on_foot_miles'] + 1e-9: errors.append(f"{fd['field_day_candidate_id']} over on-foot")
    if fd['grade_adjusted_miles'] > B['max_daily_grade_adjusted_miles'] + 1e-9: errors.append(f"{fd['field_day_candidate_id']} over grade")
    if fd['ascent_ft'] > B['max_daily_ascent_ft'] + 1e-9: errors.append(f"{fd['field_day_candidate_id']} over ascent")
    if fd['moving_p90_minutes'] > B['max_daily_moving_p90_minutes'] + 1e-9: errors.append(f"{fd['field_day_candidate_id']} over moving p90")
    if fd['p90_home_to_home_minutes'] > B['max_daily_door_to_door_p90_minutes'] + 1e-9: errors.append(f"{fd['field_day_candidate_id']} over door p90")
    if fd['loop_count'] > B['max_run_loops_per_day']: errors.append(f"{fd['field_day_candidate_id']} over loop count")
    if fd['parking_start_count'] > B['max_parking_starts_per_day']: errors.append(f"{fd['field_day_candidate_id']} over parking starts")
    for lid in fd['loop_ids']:
        cover[lid]+=1
for lid in loops:
    if cover[lid] != 1:
        errors.append(f"loop {lid} covered {cover[lid]} times")
if not optimizer.get('zero_gap_primary_certificate'):
    errors.append('optimizer did not report zero primary gap')
selected_p75=round(sum(fd['p75_home_to_home_minutes'] for fd in schedule['selected_field_days']),2)
if abs(selected_p75 - schedule['summary']['total_p75_home_to_home_minutes']) > 0.01:
    errors.append('selected p75 total does not match schedule summary')
if abs(optimizer['selected_upper_bound_minutes'] - optimizer['primary_lower_bound_minutes']) > 1e-5:
    errors.append('optimizer upper/lower bounds differ')
report={
    'verified': not errors,
    'error_count': len(errors),
    'errors': errors,
    'selected_field_day_count': len(schedule['selected_field_days']),
    'active_run_loop_count': len(loops),
    'total_p75_home_to_home_minutes': selected_p75,
    'primary_lower_bound_minutes': optimizer['primary_lower_bound_minutes'],
    'primary_optimality_gap_minutes': optimizer['primary_optimality_gap_minutes'],
}
print(json.dumps(report, indent=2, sort_keys=True))
sys.exit(1 if errors else 0)
