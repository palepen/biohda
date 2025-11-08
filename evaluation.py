"""
evaluation.py
=============
Enhanced Evaluation Module

Compares Ground Truth OTU (from TSV) with Predicted OTU (from cluster annotation)
Calculates accuracy metrics at multiple taxonomic levels

NOW USES SHARED taxonomy_parser.py FOR CONSISTENT PARSING

Usage:
    python evaluation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging
import json
import re

# Import shared taxonomy parser
from taxonomy_parser import parse_ncbi_lineage, get_taxonomy_levels, normalize_taxonomy_for_comparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedEvaluator:
    """Evaluate clustering quality using Ground Truth vs Predicted OTU"""
    
    def __init__(self,
                 predictions_file: str = "dataset/annotation/sequence_predictions.csv",
                 ground_truth_file: str = "dataset/ssu_accession_full_lineage.tsv",
                 output_dir: str = "dataset/evaluation"):
        """
        Initialize evaluator
        
        Args:
            predictions_file: CSV with seqID, cluster_id, and pred_* columns
            ground_truth_file: TSV with seqID, taxID, species, full_taxonomy
            output_dir: Directory to save evaluation results
        """
        self.predictions_file = Path(predictions_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard taxonomic levels - USE SHARED DEFINITION
        self.tax_levels = get_taxonomy_levels()
    
    def load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth taxonomy from TSV using SHARED PARSER"""
        logger.info("\nLoading ground truth taxonomy...")
        
        if not self.ground_truth_file.exists():
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_file}")
        
        gt_df = pd.read_csv(
            self.ground_truth_file,
            sep='\t',
            header=None,
            names=['seqID', 'taxID', 'species_name', 'full_taxonomy'],
            dtype=str,
            keep_default_na=False,
            na_values=['', 'N/A']
        )
        
        # Parse full taxonomy into levels using SHARED PARSER
        logger.info("  Parsing taxonomy using shared parser...")
        parsed_tax = gt_df.apply(
            lambda row: parse_ncbi_lineage(row['full_taxonomy'], row['species_name']),
            axis=1
        )
        
        # Expand into separate columns
        for level in self.tax_levels:
            gt_df[f'gt_{level}'] = parsed_tax.apply(lambda x: x.get(level))
        
        # Handle N/A
        gt_df['has_taxonomy'] = gt_df['full_taxonomy'].notna() & (gt_df['full_taxonomy'] != '')
        
        logger.info(f"  Loaded {len(gt_df):,} ground truth records")
        logger.info(f"  Records with taxonomy: {gt_df['has_taxonomy'].sum():,}")
        logger.info(f"  Records without taxonomy (N/A): {(~gt_df['has_taxonomy']).sum():,}")
        
        # Log sample of parsed taxonomy for debugging
        sample_idx = gt_df[gt_df['has_taxonomy']].index[0] if gt_df['has_taxonomy'].any() else None
        if sample_idx is not None:
            logger.info(f"\n  Sample GROUND TRUTH parsing (seqID: {gt_df.loc[sample_idx, 'seqID']}):")
            logger.info(f"    Raw lineage: {gt_df.loc[sample_idx, 'full_taxonomy'][:100]}...")
            for level in self.tax_levels:
                val = gt_df.loc[sample_idx, f'gt_{level}']
                if val:
                    logger.info(f"    gt_{level}: {val}")
        
        return gt_df
    
    def load_predictions(self) -> pd.DataFrame:
        """Load predicted OTU labels from cluster_annotation.py"""
        logger.info("\nLoading predictions...")
        
        if not self.predictions_file.exists():
            logger.error(f"Predictions file not found: {self.predictions_file}")
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_file}")
        
        pred_df = pd.read_csv(self.predictions_file)
        
        # Check if new columns exist
        if 'pred_species' not in pred_df.columns:
            logger.error("="*80)
            logger.error("ERROR: `sequence_predictions.csv` is outdated.")
            logger.error("Please re-run `cluster_annotation.py` (the new version)")
            logger.error("to generate `pred_domain`, `pred_kingdom` columns, etc.")
            logger.error("="*80)
            raise ValueError("Prediction file is missing required taxonomic columns.")
        
        logger.info(f"  Loaded {len(pred_df):,} predictions")
        logger.info(f"  Novel predictions: {pred_df['classification_type'].str.contains('novel|noise', case=False, na=False).sum():,}")
        logger.info(f"  Known predictions: {pred_df['classification_type'].str.contains('known', case=False, na=False).sum():,}")
        
        # Log sample of predicted taxonomy for debugging
        sample_idx = pred_df[pred_df['pred_genus'].notna()].index[0] if pred_df['pred_genus'].notna().any() else None
        if sample_idx is not None:
            logger.info(f"\n  Sample predicted taxonomy (seqID: {pred_df.loc[sample_idx, 'seqID']}):")
            for level in self.tax_levels:
                col = f'pred_{level}'
                if col in pred_df.columns:
                    val = pred_df.loc[sample_idx, col]
                    if pd.notna(val):
                        logger.info(f"    pred_{level}: {val}")
        
        return pred_df
    
    def create_master_table(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with ground truth"""
        logger.info("\nCreating master evaluation table...")
        
        master_df = pred_df.merge(gt_df, on='seqID', how='left')
        
        # Mark sequences with ground truth taxonomy (not N/A)
        if 'has_taxonomy' in master_df.columns:
            master_df['has_ground_truth'] = master_df['has_taxonomy'].fillna(False).astype(bool)
        else:
            master_df['has_ground_truth'] = ~master_df['gt_genus'].isna()
        
        logger.info(f"  Total sequences: {len(master_df):,}")
        logger.info(f"  With ground truth: {master_df['has_ground_truth'].sum():,}")
        logger.info(f"  Without ground truth: {(~master_df['has_ground_truth']).sum():,}")
        
        return master_df
    
    def calculate_accuracy_at_level(self, master_df: pd.DataFrame, level: str) -> Dict:
        """Calculate accuracy at a specific taxonomic level"""
        logger.info(f"\nCalculating accuracy at {level} level...")
        
        gt_col = f'gt_{level}'
        pred_col = f'pred_{level}'
        
        # Check if columns exist
        if gt_col not in master_df.columns or pred_col not in master_df.columns:
            logger.warning(f"  Missing columns for {level} level. Skipping.")
            return {
                'level': level,
                'total_sequences': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        # Filter to sequences with valid ground truth AND valid predictions at this level
        eval_df = master_df[
            master_df['has_ground_truth'] & 
            master_df[gt_col].notna() & 
            master_df[pred_col].notna()
        ].copy()
        
        if len(eval_df) == 0:
            logger.warning(f"  No sequences with both ground truth and predictions at {level} level")
            return {
                'level': level,
                'total_sequences': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        # Normalize strings for comparison using SHARED FUNCTION
        eval_df[f'{gt_col}_norm'] = eval_df[gt_col].apply(normalize_taxonomy_for_comparison)
        eval_df[f'{pred_col}_norm'] = eval_df[pred_col].apply(normalize_taxonomy_for_comparison)
        
        # Compare
        matches = eval_df[f'{gt_col}_norm'] == eval_df[f'{pred_col}_norm']
        correct = matches.sum()
        incorrect = len(eval_df) - correct
        accuracy = correct / len(eval_df)
        
        logger.info(f"  Total sequences: {len(eval_df):,}")
        logger.info(f"  Correct: {correct:,}")
        logger.info(f"  Incorrect: {incorrect:,}")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        
        # Log sample mismatches for debugging (always show at least 10)
        if incorrect > 0:
            num_samples = min(10, incorrect)
            logger.info(f"\n  Sample mismatches (showing {num_samples}):")
            mismatches = eval_df[~matches].head(num_samples)
            for idx, row in mismatches.iterrows():
                logger.info(f"    {row['seqID']}:")
                logger.info(f"      GT:   {row[gt_col]}")
                logger.info(f"      PRED: {row[pred_col]}")
        
        # Also show some correct matches for comparison
        if correct > 0:
            num_samples = min(5, correct)
            logger.info(f"\n  Sample correct matches (showing {num_samples}):")
            correct_matches = eval_df[matches].head(num_samples)
            for idx, row in correct_matches.iterrows():
                logger.info(f"    {row['seqID']}: {row[gt_col]} ✓")
        
        return {
            'level': level,
            'total_sequences': len(eval_df),
            'correct': int(correct),
            'incorrect': int(incorrect),
            'accuracy': float(accuracy)
        }
    
    def evaluate_novelty_detection(self, master_df: pd.DataFrame) -> Dict:
        """
        Evaluate novelty detection performance
        
        True Positive: Ground Truth = N/A (no taxonomy), Predicted = Novel
        False Positive: Ground Truth = Known, Predicted = Novel
        True Negative: Ground Truth = Known, Predicted = Known (correct match at genus)
        False Negative: Ground Truth = N/A (no taxonomy), Predicted = Known
        """
        logger.info("\nEvaluating novelty detection...")
        
        # Define Novel predictions (includes Noise and Unknown)
        is_predicted_novel = (
            master_df['classification_type'].str.contains('novel', case=False, na=False) |
            master_df['classification_type'].str.contains('noise', case=False, na=False) |
            master_df['pred_species'].isna()
        )
        
        # Define Known ground truth (has taxonomy information)
        has_ground_truth = master_df['has_ground_truth']
        
        # Check for correct genus match for True Negatives
        gt_genus_norm = master_df['gt_genus'].apply(normalize_taxonomy_for_comparison)
        pred_genus_norm = master_df['pred_genus'].apply(normalize_taxonomy_for_comparison)
        is_correct_genus = (gt_genus_norm == pred_genus_norm)
        is_correct_genus = is_correct_genus.fillna(False)
        
        # Calculate confusion matrix
        tp = ((~has_ground_truth) & is_predicted_novel).sum()  # Novel correctly identified
        fp = (has_ground_truth & is_predicted_novel).sum()      # Known wrongly called novel
        tn = (has_ground_truth & ~is_predicted_novel & is_correct_genus).sum() # Known correctly identified
        fn = ((~has_ground_truth) & ~is_predicted_novel).sum()  # Novel wrongly called known
        
        total = len(master_df)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        
        logger.info(f"  True Positives (Novel→Novel): {tp:,}")
        logger.info(f"  False Positives (Known→Novel): {fp:,}")
        logger.info(f"  True Negatives (Known→Known, correct genus): {tn:,}")
        logger.info(f"  False Negatives (Novel→Known): {fn:,}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1_score:.3f}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'accuracy': float(accuracy)
        }
    
    def analyze_misclassifications(self, master_df: pd.DataFrame, level: str = 'genus') -> Dict:
        """Analyze common misclassification patterns at a given level"""
        logger.info(f"\nAnalyzing misclassifications at {level} level...")
        
        gt_col = f'gt_{level}'
        pred_col = f'pred_{level}'
        
        # Focus on sequences with ground truth that were misclassified
        eval_df = master_df[
            master_df['has_ground_truth'] & 
            master_df[gt_col].notna() & 
            master_df[pred_col].notna()
        ].copy()
        
        if len(eval_df) == 0:
            logger.info("  No sequences available for misclassification analysis")
            return {'total_misclassified': 0}
        
        # Normalize for comparison
        eval_df[f'{gt_col}_norm'] = eval_df[gt_col].apply(normalize_taxonomy_for_comparison)
        eval_df[f'{pred_col}_norm'] = eval_df[pred_col].apply(normalize_taxonomy_for_comparison)
        
        misclassified = eval_df[eval_df[f'{gt_col}_norm'] != eval_df[f'{pred_col}_norm']].copy()
        
        if len(misclassified) == 0:
            logger.info("  No misclassifications found")
            return {
                'total_misclassified': 0,
                'misclassification_rate': 0.0,
                'top_patterns': []
            }
        
        # Common misclassification pairs
        misclass_pairs = Counter(
            zip(misclassified[gt_col], misclassified[pred_col])
        )
        
        top_10_pairs = misclass_pairs.most_common(10)
        
        logger.info(f"  Total misclassified: {len(misclassified):,}")
        logger.info(f"  Misclassification rate: {len(misclassified)/len(eval_df):.2%}")
        logger.info(f"  Top 10 misclassification patterns:")
        for (gt, pred), count in top_10_pairs:
            logger.info(f"    {gt} -> {pred}: {count}")
        
        return {
            'level': level,
            'total_misclassified': len(misclassified),
            'total_evaluated': len(eval_df),
            'misclassification_rate': len(misclassified) / len(eval_df),
            'top_patterns': [{'ground_truth': gt, 'predicted': pred, 'count': count} 
                           for (gt, pred), count in top_10_pairs]
        }
    
    def calculate_cluster_purity(self, master_df: pd.DataFrame, level: str = 'genus') -> Dict:
        """Calculate taxonomic purity of clusters at a given level"""
        logger.info(f"\nCalculating cluster purity at {level} level...")
        
        gt_col = f'gt_{level}'
        cluster_purity = []
        
        for cluster_id in master_df['cluster_id'].unique():
            if cluster_id == -1:
                continue
            
            cluster_data = master_df[
                (master_df['cluster_id'] == cluster_id) & 
                master_df['has_ground_truth'] &
                master_df[gt_col].notna()
            ]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate purity
            tax_counts = cluster_data[gt_col].value_counts()
            if len(tax_counts) > 0:
                most_common_count = tax_counts.iloc[0]
                purity = most_common_count / len(cluster_data)
                
                cluster_purity.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'purity': purity,
                    f'dominant_{level}': tax_counts.index[0],
                    f'num_{level}': len(tax_counts)
                })
        
        if cluster_purity:
            purity_df = pd.DataFrame(cluster_purity)
            mean_purity = purity_df['purity'].mean()
            median_purity = purity_df['purity'].median()
            
            logger.info(f"  Clusters evaluated: {len(purity_df)}")
            logger.info(f"  Mean cluster purity: {mean_purity:.3f}")
            logger.info(f"  Median cluster purity: {median_purity:.3f}")
            logger.info(f"  High purity clusters (>0.9): {(purity_df['purity'] > 0.9).sum()}")
            logger.info(f"  Medium purity clusters (0.7-0.9): {((purity_df['purity'] > 0.7) & (purity_df['purity'] <= 0.9)).sum()}")
            logger.info(f"  Low purity clusters (<0.7): {(purity_df['purity'] <= 0.7).sum()}")
            
            return {
                'level': level,
                'mean_purity': float(mean_purity),
                'median_purity': float(median_purity),
                'high_purity_clusters': int((purity_df['purity'] > 0.9).sum()),
                'medium_purity_clusters': int(((purity_df['purity'] > 0.7) & (purity_df['purity'] <= 0.9)).sum()),
                'low_purity_clusters': int((purity_df['purity'] <= 0.7).sum()),
                'total_clusters': len(purity_df),
                'purity_distribution': purity_df['purity'].describe().to_dict()
            }
        
        logger.info(f"  No clusters with sufficient ground truth data for {level} purity calculation")
        return {'level': level, 'mean_purity': 0.0, 'total_clusters': 0}
    
    def generate_report(self, master_df: pd.DataFrame, accuracy_results: List[Dict], 
                       novelty_results: Dict, misclass_results: Dict, purity_results: Dict):
        """Generate comprehensive evaluation report"""
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING EVALUATION REPORT")
        logger.info(f"{'='*80}")
        
        report_file = self.output_dir / "evaluation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED EVALUATION REPORT\n")
            f.write("Ground Truth vs Predicted OTU Comparison (Multi-Level)\n")
            f.write("Using Shared taxonomy_parser.py for consistent parsing\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total sequences: {len(master_df):,}\n")
            f.write(f"Sequences with ground truth: {master_df['has_ground_truth'].sum():,}\n")
            f.write(f"Sequences without ground truth (N/A): {(~master_df['has_ground_truth']).sum():,}\n")
            f.write(f"Total clusters: {master_df['cluster_id'].nunique()}\n")
            f.write(f"Noise sequences (cluster -1): {(master_df['cluster_id'] == -1).sum():,}\n\n")
            
            # Accuracy at each level
            f.write("="*80 + "\n")
            f.write("1. ACCURACY AT TAXONOMIC LEVELS\n")
            f.write("="*80 + "\n\n")
            
            for result in accuracy_results:
                f.write(f"{result['level'].upper()} Level:\n")
                if result['total_sequences'] > 0:
                    f.write(f"  Total sequences evaluated: {result['total_sequences']:,}\n")
                    f.write(f"  Correct predictions: {result['correct']:,}\n")
                    f.write(f"  Incorrect predictions: {result['incorrect']:,}\n")
                    f.write(f"  Accuracy: {result['accuracy']:.2%}\n\n")
                else:
                    f.write(f"  No data available for evaluation at this level\n\n")
            
            # Novelty detection
            f.write("="*80 + "\n")
            f.write("2. NOVELTY DETECTION PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"True Positives (Novel→Novel): {novelty_results['true_positives']:,}\n")
            f.write(f"False Positives (Known→Novel): {novelty_results['false_positives']:,}\n")
            f.write(f"True Negatives (Known→Known, correct genus): {novelty_results['true_negatives']:,}\n")
            f.write(f"False Negatives (Novel→Known): {novelty_results['false_negatives']:,}\n\n")
            
            f.write(f"Precision: {novelty_results['precision']:.3f}\n")
            f.write(f"Recall: {novelty_results['recall']:.3f}\n")
            f.write(f"F1-Score: {novelty_results['f1_score']:.3f}\n")
            f.write(f"Accuracy: {novelty_results['accuracy']:.3f}\n\n")
            
            # Cluster purity
            f.write("="*80 + "\n")
            f.write("3. CLUSTER QUALITY (TAXONOMIC PURITY)\n")
            f.write("="*80 + "\n\n")
            
            if purity_results.get('total_clusters', 0) > 0:
                f.write(f"Purity calculated at {purity_results.get('level', 'N/A')} level\n")
                f.write(f"Clusters evaluated: {purity_results['total_clusters']}\n")
                f.write(f"Mean purity: {purity_results.get('mean_purity', 0):.3f}\n")
                f.write(f"Median purity: {purity_results.get('median_purity', 0):.3f}\n")
                f.write(f"High purity clusters (>0.9): {purity_results.get('high_purity_clusters', 0)}\n")
                f.write(f"Medium purity clusters (0.7-0.9): {purity_results.get('medium_purity_clusters', 0)}\n")
                f.write(f"Low purity clusters (<0.7): {purity_results.get('low_purity_clusters', 0)}\n\n")
            else:
                f.write("No clusters with sufficient ground truth data\n\n")
            
            # Misclassifications
            f.write("="*80 + "\n")
            f.write("4. MISCLASSIFICATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            if misclass_results.get('total_evaluated', 0) > 0:
                f.write(f"Analysis at {misclass_results.get('level', 'N/A')} level\n")
                f.write(f"Total sequences evaluated: {misclass_results['total_evaluated']:,}\n")
                f.write(f"Total misclassified: {misclass_results.get('total_misclassified', 0):,}\n")
                f.write(f"Misclassification rate: {misclass_results.get('misclassification_rate', 0):.2%}\n\n")
                
                if misclass_results.get('top_patterns'):
                    f.write("Top 10 misclassification patterns (Genus level):\n")
                    for pattern in misclass_results['top_patterns']:
                        f.write(f"  {pattern['ground_truth']} -> {pattern['predicted']}: {pattern['count']}\n")
                    f.write("\n")
            else:
                f.write("No misclassification data available\n\n")
            
            # Overall assessment
            f.write("="*80 + "\n")
            f.write("5. OVERALL ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            # Calculate average accuracy from genus and species
            species_acc = next((r['accuracy'] for r in accuracy_results if r['level'] == 'species'), 0.0)
            genus_acc = next((r['accuracy'] for r in accuracy_results if r['level'] == 'genus'), 0.0)
            
            f.write(f"Genus Accuracy: {genus_acc:.2%}\n")
            f.write(f"Species Accuracy: {species_acc:.2%}\n")
            f.write(f"Novelty Detection F1-Score: {novelty_results['f1_score']:.3f}\n")
            f.write(f"Cluster Purity (Mean, Genus): {purity_results.get('mean_purity', 0):.3f}\n\n")
            
            # Performance rating
            if species_acc >= 0.8 and novelty_results['f1_score'] >= 0.7:
                rating = "EXCELLENT: High agreement with ground truth, strong novelty detection"
            elif genus_acc >= 0.7 and novelty_results['f1_score'] >= 0.5:
                rating = "GOOD: Strong genus-level accuracy"
            elif genus_acc >= 0.5 or novelty_results['f1_score'] >= 0.4:
                rating = "FAIR: Moderate performance, consider parameter tuning"
            else:
                rating = "POOR: Low performance, significant improvements needed"
            
            f.write(f"Overall Rating: {rating}\n\n")
            
            # Recommendations
            f.write("="*80 + "\n")
            f.write("6. RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            if species_acc < 0.7:
                f.write("- Species accuracy is low. Check `master_evaluation_table.csv` for common errors.\n")
            
            if novelty_results['f1_score'] < 0.5:
                f.write("- Tune novelty detection thresholds (identity_threshold_novel)\n")
                f.write("- Review False Positives (Known→Novel) and False Negatives (Novel→Known)\n")
            
            if purity_results.get('mean_purity', 0) < 0.7:
                f.write("- Adjust clustering parameters (min_cluster_size, min_samples)\n")
                f.write("- Consider alternative distance metrics or dimensionality reduction\n")
            
            if (master_df['cluster_id'] == -1).sum() / len(master_df) > 0.5:
                f.write(f"- High noise ratio ({(master_df['cluster_id'] == -1).sum() / len(master_df):.1%}).\n")
                f.write("- Adjust HDBSCAN `min_cluster_size` or `min_samples` in `clustering.py`.\n")

        
        logger.info(f"  Report saved: {report_file}")
        
        # Save master table
        master_file = self.output_dir / "master_evaluation_table.csv"
        master_df.to_csv(master_file, index=False)
        logger.info(f"  Master table saved: {master_file}")
        
        # Save metrics as JSON
        metrics = {
            'dataset_overview': {
                'total_sequences': len(master_df),
                'sequences_with_ground_truth': int(master_df['has_ground_truth'].sum()),
                'sequences_without_ground_truth': int((~master_df['has_ground_truth']).sum())
            },
            'accuracy_by_level': accuracy_results,
            'novelty_detection': novelty_results,
            'cluster_purity_genus': purity_results,
            'misclassifications_genus': misclass_results
        }
        
        metrics_file = self.output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Metrics saved: {metrics_file}")
    
    def run_evaluation(self):
        """Execute complete evaluation pipeline"""
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED EVALUATION PIPELINE (Multi-Level)")
        logger.info("Using Shared taxonomy_parser.py for Consistent Parsing")
        logger.info(f"{'='*80}\n")
        
        try:
            # Load data
            gt_df = self.load_ground_truth()
            pred_df = self.load_predictions()
            
            # Create master table
            master_df = self.create_master_table(pred_df, gt_df)
            
            # Calculate metrics
            accuracy_results = []
            for level in self.tax_levels:
                result = self.calculate_accuracy_at_level(master_df, level)
                accuracy_results.append(result)
            
            novelty_results = self.evaluate_novelty_detection(master_df)
            misclass_results_genus = self.analyze_misclassifications(master_df, level='genus')
            purity_results_genus = self.calculate_cluster_purity(master_df, level='genus')
            
            # Generate report
            self.generate_report(master_df, accuracy_results, novelty_results, 
                               misclass_results_genus, purity_results_genus)
            
            logger.info(f"\n{'='*80}")
            logger.info("EVALUATION COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"\nResults saved to: {self.output_dir}")
            return 0

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 1
        except ValueError as e:
            logger.error(f"DataError: {e}")
            return 1
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    evaluator = EnhancedEvaluator(
        predictions_file="dataset/annotation/sequence_predictions.csv",
        ground_truth_file="dataset/ssu_accession_full_lineage.tsv",
        output_dir="dataset/evaluation"
    )
    
    return evaluator.run_evaluation()


if __name__ == "__main__":
    exit(main())