"""
Database Query Examples - Enhanced Clinical Trial Extractor
Demonstrates powerful querying capabilities for multiple timepoints and Cochrane-compliant data extraction
"""

from app import app, db, Study, Outcome, OutcomeTimepoint, Intervention
from sqlalchemy import and_, or_, func
import json

def demonstrate_queries():
    """Demonstrate advanced querying capabilities"""
    
    with app.app_context():
        print("=== ENHANCED DATABASE QUERYING CAPABILITIES ===\n")
        
        # 1. Find all studies with 12-week primary outcomes
        print("1. Studies with 12-week primary outcomes:")
        studies_12w = db.session.query(Study).join(Outcome).join(OutcomeTimepoint).filter(
            and_(
                OutcomeTimepoint.timepoint_value == 12,
                OutcomeTimepoint.timepoint_unit == 'weeks',
                Outcome.outcome_type == 'primary'
            )
        ).all()
        
        for study in studies_12w:
            print(f"   - {study.title[:50]}...")
        print()
        
        # 2. Find outcomes measured at multiple timepoints
        print("2. Outcomes with multiple timepoints:")
        outcomes_multi_tp = db.session.query(Outcome).join(OutcomeTimepoint).group_by(Outcome.id).having(
            func.count(OutcomeTimepoint.id) > 1
        ).all()
        
        for outcome in outcomes_multi_tp:
            timepoints = [f"{tp.timepoint_value} {tp.timepoint_unit}" for tp in outcome.timepoints]
            print(f"   - {outcome.outcome_name[:40]}... measured at: {', '.join(timepoints)}")
        print()
        
        # 3. Find all secondary outcomes with significant p-values
        print("3. Secondary outcomes with p < 0.05:")
        significant_outcomes = db.session.query(OutcomeTimepoint).join(Outcome).filter(
            and_(
                Outcome.outcome_type == 'secondary',
                OutcomeTimepoint.p_value < 0.05
            )
        ).all()
        
        for tp in significant_outcomes:
            print(f"   - {tp.outcome.outcome_name[:40]}... at {tp.timepoint_name}: p={tp.p_value}")
        print()
        
        # 4. Compare short-term vs long-term effects
        print("4. Short-term (≤4 weeks) vs Long-term (≥12 weeks) effects:")
        short_term = db.session.query(OutcomeTimepoint).filter(
            and_(
                OutcomeTimepoint.timepoint_unit == 'weeks',
                OutcomeTimepoint.timepoint_value <= 4
            )
        ).count()
        
        long_term = db.session.query(OutcomeTimepoint).filter(
            and_(
                OutcomeTimepoint.timepoint_unit == 'weeks', 
                OutcomeTimepoint.timepoint_value >= 12
            )
        ).count()
        
        print(f"   - Short-term timepoints (≤4 weeks): {short_term}")
        print(f"   - Long-term timepoints (≥12 weeks): {long_term}")
        print()
        
        # 5. Find studies with interim analyses
        print("5. Studies with interim analyses:")
        interim_studies = db.session.query(Study).join(Outcome).join(OutcomeTimepoint).filter(
            OutcomeTimepoint.timepoint_type == 'interim'
        ).distinct().all()
        
        for study in interim_studies:
            print(f"   - {study.title[:50]}...")
        print()
        
        # 6. Query by effect size and confidence intervals
        print("6. Large effect sizes (>1.0) with tight confidence intervals:")
        large_effects = db.session.query(OutcomeTimepoint).filter(
            and_(
                OutcomeTimepoint.effect_estimate > 1.0,
                (OutcomeTimepoint.effect_ci_upper - OutcomeTimepoint.effect_ci_lower) < 1.0
            )
        ).all()
        
        for tp in large_effects:
            ci_width = tp.effect_ci_upper - tp.effect_ci_lower if tp.effect_ci_upper and tp.effect_ci_lower else "N/A"
            print(f"   - {tp.outcome.outcome_name[:30]}...: Effect={tp.effect_estimate}, CI width={ci_width}")
        print()
        
        # 7. Meta-analysis ready data extraction
        print("7. Meta-analysis ready data for blood pressure outcomes:")
        bp_outcomes = db.session.query(OutcomeTimepoint).join(Outcome).filter(
            or_(
                Outcome.outcome_name.ilike('%blood pressure%'),
                Outcome.outcome_name.ilike('%systolic%'),
                Outcome.outcome_name.ilike('%diastolic%')
            )
        ).all()
        
        for tp in bp_outcomes:
            print(f"   - Study {tp.study_id}: {tp.outcome.outcome_name[:30]}... at {tp.timepoint_name}")
            print(f"     Effect: {tp.effect_estimate} ({tp.effect_ci_lower} to {tp.effect_ci_upper}), p={tp.p_value}")
        print()
        
        # 8. Source tracking and data quality
        print("8. High-confidence data sources:")
        high_conf = db.session.query(OutcomeTimepoint).filter(
            OutcomeTimepoint.source_confidence == 'high'
        ).count()
        
        total_timepoints = db.session.query(OutcomeTimepoint).count()
        print(f"   - High-confidence timepoints: {high_conf}/{total_timepoints} ({high_conf/total_timepoints*100:.1f}%)")
        print()
        
        # 9. Complex Cochrane-style queries
        print("9. Cochrane Review Query - Primary outcomes at primary timepoints:")
        primary_at_primary = db.session.query(OutcomeTimepoint).join(Outcome).filter(
            and_(
                Outcome.outcome_type == 'primary',
                OutcomeTimepoint.timepoint_type == 'primary'
            )
        ).all()
        
        for tp in primary_at_primary:
            print(f"   - {tp.outcome.outcome_name[:40]}... at {tp.timepoint_name}")
            print(f"     N={tp.n_analyzed}, Mean={tp.mean_value}, SD={tp.sd_value}")
            print(f"     Source: {tp.data_source}")
        print()
        
        print("=== SUMMARY ===")
        print(f"Total studies: {db.session.query(Study).count()}")
        print(f"Total outcomes: {db.session.query(Outcome).count()}")
        print(f"Total timepoints: {db.session.query(OutcomeTimepoint).count()}")
        print(f"Studies with multiple timepoints: {len(outcomes_multi_tp)}")

def export_for_meta_analysis(outcome_pattern=""):
    """Export data in format ready for meta-analysis"""
    
    with app.app_context():
        # Query for specific outcome across all studies and timepoints
        query = db.session.query(
            Study.title.label('study'),
            Study.year.label('year'),
            Outcome.outcome_name.label('outcome'),
            OutcomeTimepoint.timepoint_name.label('timepoint'),
            OutcomeTimepoint.timepoint_value.label('timepoint_value'),
            OutcomeTimepoint.timepoint_unit.label('timepoint_unit'),
            OutcomeTimepoint.n_analyzed.label('n'),
            OutcomeTimepoint.mean_value.label('mean'),
            OutcomeTimepoint.sd_value.label('sd'),
            OutcomeTimepoint.effect_estimate.label('effect_estimate'),
            OutcomeTimepoint.effect_ci_lower.label('ci_lower'),
            OutcomeTimepoint.effect_ci_upper.label('ci_upper'),
            OutcomeTimepoint.p_value.label('p_value'),
            OutcomeTimepoint.data_source.label('source')
        ).select_from(Study).join(Outcome).join(OutcomeTimepoint)
        
        if outcome_pattern:
            query = query.filter(Outcome.outcome_name.ilike(f'%{outcome_pattern}%'))
        
        results = query.all()
        
        print(f"\n=== META-ANALYSIS EXPORT: {outcome_pattern.upper() if outcome_pattern else 'ALL OUTCOMES'} ===")
        print("Study\tYear\tOutcome\tTimepoint\tN\tMean\tSD\tEffect\tCI_Lower\tCI_Upper\tP_Value\tSource")
        
        for result in results:
            print(f"{result.study[:30]}\t{result.year}\t{result.outcome[:30]}\t{result.timepoint}\t"
                  f"{result.n}\t{result.mean}\t{result.sd}\t{result.effect_estimate}\t"
                  f"{result.ci_lower}\t{result.ci_upper}\t{result.p_value}\t{result.source[:50]}")

if __name__ == "__main__":
    demonstrate_queries()
    export_for_meta_analysis("blood pressure")