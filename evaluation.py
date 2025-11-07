"""
evaluation.py
=============
Enhanced Evaluation Module

Compares Ground Truth OTU (from TSV) with Predicted OTU (from cluster annotation)
Calculates accuracy metrics at multiple taxonomic levels

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
            predictions_file: CSV with seqID, cluster_id, predicted_otu
            ground_truth_file: TSV with seqID, taxID, species, full_taxonomy
            output_dir: Directory to save evaluation results
        """
        self.predictions_file = Path(predictions_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tax_levels = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    def load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth taxonomy from TSV"""
        logger.info("\nLoading ground truth taxonomy...")
        
        gt_df = pd.read_csv(
            self.ground_truth_file,
            sep='\t',
            header=None,
            names=['seqID', 'taxID', 'species', 'full_taxonomy'],
            dtype=str,
            keep_default_na=False
        )
        
        # Parse full taxonomy into levels
        gt_df['parsed_taxonomy'] = gt_df['full_taxonomy'].apply(self._parse_taxonomy)
        
        # Expand into separate columns
        for level in self.tax_levels:
            gt_df[f'gt_{level}'] = gt_df['parsed_taxonomy'].apply(lambda x: x.get(level))
        
        gt_df = gt_df.drop(columns=['parsed_taxonomy'])
        
        logger.info(f"  Loaded {len(gt_df):,} ground truth records")
        
        return gt_df
    
    def _parse_taxonomy(self, tax_string: str) -> Dict[str, str]:
        """Parse semicolon-delimited taxonomy string"""
        if pd.isna(tax_string) or tax_string == '':
            return {level: None for level in self.tax_levels}
        
        parts = [p.strip() for p in tax_string.split(';') if p.strip()]
        parsed = {}
        
        for i, part in enumerate(parts):
            if i < len(self.tax_levels):
                parsed[self.tax_levels[i]] = part
        
        # Fill missing levels
        for level in self.tax_levels:
            parsed.setdefault(level, None)
        
        return parsed
    
    def load_predictions(self) -> pd.DataFrame:
        """Load predicted OTU labels"""
        logger.info("\nLoading predictions...")
        
        pred_df = pd.read_csv(self.predictions_file)
        
        # Parse predicted_otu
        pred_df['pred_genus'], pred_df['pred_species'] = zip(*pred_df['predicted_otu'].apply(self._parse_predicted_otu))
        
        logger.info(f"  Loaded {len(pred_df):,} predictions")
        
        return pred_df
    
    def _parse_predicted_otu(self, otu_string: str) -> Tuple[str, str]:
        """
        Parse predicted OTU string
        Examples:
          "Genus, Species" -> ("Genus", "Species")
          "Novel (BLAST-Divergent)" -> (None, None)
        """
        if pd.isna(otu_string) or 'Novel' in otu_string or 'Unknown' in otu_string:
            return (None, None)
        
        parts = [p.strip() for p in otu_string.split(',')]
        genus = parts[0] if len(parts) > 0 else None
        species = parts[1] if len(parts) > 1 else None
        
        return (genus, species)
    
    def create_master_table(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
        """Merge predictions with ground truth"""
        logger.info("\nCreating master evaluation table...")
        
        master_df = pred_df.merge(gt_df, on='seqID', how='left')
        
        # Mark sequences with/without ground truth
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
        
        # Filter to sequences with ground truth
        eval_df = master_df[master_df['has_ground_truth'] & master_df[gt_col].notna()].copy()
        
        if len(eval_df) == 0:
            logger.warning(f"  No sequences with ground truth at {level} level")
            return {
                'level': level,
                'total_sequences': 0,
                'correct': 0,
                'accuracy': 0.0
            }
        
        # Compare
        matches = eval_df[gt_col] == eval_df[pred_col]
        correct = matches.sum()
        accuracy = correct / len(eval_df)
        
        logger.info(f"  Total sequences: {len(eval_df):,}")
        logger.info(f"  Correct: {correct:,}")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        
        return {
            'level': level,
            'total_sequences': len(eval_df),
            'correct': int(correct),
            'accuracy': float(accuracy)
        }
    
    def evaluate_novelty_detection(self, master_df: pd.DataFrame) -> Dict:
        """
        Evaluate novelty detection performance
        
        True Positive: Ground Truth = NA, Predicted = Novel
        False Positive: Ground Truth = Known, Predicted = Novel
        True Negative: Ground Truth = Known, Predicted = Known (correct)
        False Negative: Ground Truth = NA, Predicted = Known
        """
        logger.info("\nEvaluating novelty detection...")
        
        # Define Novel predictions
        is_predicted_novel = (
            master_df['predicted_otu'].str.contains('Novel', na=False) |
            master_df['predicted_otu'].str.contains('Unknown', na=False)
        )
        
        # Define Known ground truth
        has_ground_truth = master_df['has_ground_truth']
        
        # Calculate confusion matrix
        tp = ((~has_ground_truth) & is_predicted_novel).sum()
        fp = (has_ground_truth & is_predicted_novel).sum()
        tn = (has_ground_truth & ~is_predicted_novel & (master_df['gt_genus'] == master_df['pred_genus'])).sum()
        fn = ((~has_ground_truth) & ~is_predicted_novel).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(master_df) if len(master_df) > 0 else 0
        
        logger.info(f"  True Positives: {tp:,}")
        logger.info(f"  False Positives: {fp:,}")
        logger.info(f"  True Negatives: {tn:,}")
        logger.info(f"  False Negatives: {fn:,}")
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
    
    def analyze_misclassifications(self, master_df: pd.DataFrame) -> Dict:
        """Analyze common misclassification patterns"""
        logger.info("\nAnalyzing misclassifications...")
        
        # Focus on sequences with ground truth that were misclassified
        misclassified = master_df[
            master_df['has_ground_truth'] & 
            (master_df['gt_genus'] != master_df['pred_genus'])
        ].copy()
        
        if len(misclassified) == 0:
            logger.info("  No misclassifications found")
            return {'total_misclassified': 0}
        
        # Common misclassification pairs
        misclass_pairs = Counter(
            zip(misclassified['gt_genus'].fillna('NA'), 
                misclassified['pred_genus'].fillna('Novel'))
        )
        
        top_10_pairs = misclass_pairs.most_common(10)
        
        logger.info(f"  Total misclassified: {len(misclassified):,}")
        logger.info(f"  Top 10 misclassification patterns:")
        for (gt, pred), count in top_10_pairs:
            logger.info(f"    {gt} -> {pred}: {count}")
        
        return {
            'total_misclassified': len(misclassified),
            'misclassification_rate': len(misclassified) / master_df['has_ground_truth'].sum(),
            'top_patterns': [{'ground_truth': gt, 'predicted': pred, 'count': count} 
                           for (gt, pred), count in top_10_pairs]
        }
    
    def calculate_cluster_purity(self, master_df: pd.DataFrame) -> Dict:
        """Calculate taxonomic purity of clusters"""
        logger.info("\nCalculating cluster purity...")
        
        cluster_purity = []
        
        for cluster_id in master_df['cluster_id'].unique():
            if cluster_id == -1:
                continue
            
            cluster_data = master_df[
                (master_df['cluster_id'] == cluster_id) & 
                master_df['has_ground_truth']
            ]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate genus-level purity
            genus_counts = cluster_data['gt_genus'].value_counts()
            if len(genus_counts) > 0:
                most_common_count = genus_counts.iloc[0]
                purity = most_common_count / len(cluster_data)
                
                cluster_purity.append({
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'purity': purity,
                    'dominant_genus': genus_counts.index[0]
                })
        
        if cluster_purity:
            purity_df = pd.DataFrame(cluster_purity)
            mean_purity = purity_df['purity'].mean()
            
            logger.info(f"  Mean cluster purity: {mean_purity:.3f}")
            logger.info(f"  High purity clusters (>0.8): {(purity_df['purity'] > 0.8).sum()}")
            
            return {
                'mean_purity': float(mean_purity),
                'median_purity': float(purity_df['purity'].median()),
                'high_purity_clusters': int((purity_df['purity'] > 0.8).sum()),
                'total_clusters': len(purity_df)
            }
        
        return {'mean_purity': 0.0}
    
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
            f.write("Ground Truth vs Predicted OTU Comparison\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total sequences: {len(master_df):,}\n")
            f.write(f"Sequences with ground truth: {master_df['has_ground_truth'].sum():,}\n")
            f.write(f"Sequences without ground truth: {(~master_df['has_ground_truth']).sum():,}\n")
            f.write(f"Total clusters: {master_df['cluster_id'].nunique()}\n\n")
            
            # Accuracy at each level
            f.write("="*80 + "\n")
            f.write("1. ACCURACY AT TAXONOMIC LEVELS\n")
            f.write("="*80 + "\n\n")
            
            for result in accuracy_results:
                f.write(f"{result['level'].upper()} Level:\n")
                f.write(f"  Total sequences: {result['total_sequences']:,}\n")
                f.write(f"  Correct predictions: {result['correct']:,}\n")
                f.write(f"  Accuracy: {result['accuracy']:.2%}\n\n")
            
            # Novelty detection
            f.write("="*80 + "\n")
            f.write("2. NOVELTY DETECTION PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"True Positives (Novel detected as novel): {novelty_results['true_positives']:,}\n")
            f.write(f"False Positives (Known detected as novel): {novelty_results['false_positives']:,}\n")
            f.write(f"True Negatives (Known detected as known): {novelty_results['true_negatives']:,}\n")
            f.write(f"False Negatives (Novel detected as known): {novelty_results['false_negatives']:,}\n\n")
            
            f.write(f"Precision: {novelty_results['precision']:.3f}\n")
            f.write(f"Recall: {novelty_results['recall']:.3f}\n")
            f.write(f"F1-Score: {novelty_results['f1_score']:.3f}\n")
            f.write(f"Accuracy: {novelty_results['accuracy']:.3f}\n\n")
            
            # Cluster purity
            f.write("="*80 + "\n")
            f.write("3. CLUSTER QUALITY (PURITY)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Mean purity: {purity_results.get('mean_purity', 0):.3f}\n")
            f.write(f"Median purity: {purity_results.get('median_purity', 0):.3f}\n")
            f.write(f"High purity clusters (>0.8): {purity_results.get('high_purity_clusters', 0)}\n\n")
            
            # Misclassifications
            f.write("="*80 + "\n")
            f.write("4. MISCLASSIFICATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total misclassified: {misclass_results.get('total_misclassified', 0):,}\n")
            if 'misclassification_rate' in misclass_results:
                f.write(f"Misclassification rate: {misclass_results['misclassification_rate']:.2%}\n\n")
            
            if 'top_patterns' in misclass_results:
                f.write("Top misclassification patterns:\n")
                for pattern in misclass_results['top_patterns'][:10]:
                    f.write(f"  {pattern['ground_truth']} -> {pattern['predicted']}: {pattern['count']}\n")
                f.write("\n")
            
            # Overall assessment
            f.write("="*80 + "\n")
            f.write("5. OVERALL ASSESSMENT\n")
            f.write("="*80 + "\n\n")
            
            avg_accuracy = np.mean([r['accuracy'] for r in accuracy_results if r['total_sequences'] > 0])
            
            f.write(f"Average Accuracy (Genus/Species): {avg_accuracy:.2%}\n")
            f.write(f"Novelty Detection F1-Score: {novelty_results['f1_score']:.3f}\n")
            f.write(f"Cluster Purity: {purity_results.get('mean_purity', 0):.3f}\n\n")
            
            # Performance rating
            if avg_accuracy >= 0.8 and novelty_results['f1_score'] >= 0.7:
                f.write("EXCELLENT: High agreement with ground truth and strong novelty detection\n")
            elif avg_accuracy >= 0.6 and novelty_results['f1_score'] >= 0.5:
                f.write("GOOD: Reasonable performance across metrics\n")
            elif avg_accuracy >= 0.4 or novelty_results['f1_score'] >= 0.4:
                f.write("FAIR: Moderate performance, consider parameter tuning\n")
            else:
                f.write("POOR: Low performance, significant improvements needed\n")
        
        logger.info(f"  Report saved: {report_file}")
        
        # Save master table
        master_file = self.output_dir / "master_evaluation_table.csv"
        master_df.to_csv(master_file, index=False)
        logger.info(f"  Master table saved: {master_file}")
        
        # Save metrics as JSON
        metrics = {
            'accuracy_by_level': accuracy_results,
            'novelty_detection': novelty_results,
            'cluster_purity': purity_results,
            'misclassifications': misclass_results
        }
        
        metrics_file = self.output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Metrics saved: {metrics_file}")
    
    def run_evaluation(self):
        """Execute complete evaluation pipeline"""
        logger.info(f"\n{'='*80}")
        logger.info("ENHANCED EVALUATION PIPELINE")
        logger.info(f"{'='*80}\n")
        
        # Load data
        gt_df = self.load_ground_truth()
        pred_df = self.load_predictions()
        
        # Create master table
        master_df = self.create_master_table(pred_df, gt_df)
        
        # Calculate metrics
        accuracy_results = []
        for level in ['genus', 'species']:
            result = self.calculate_accuracy_at_level(master_df, level)
            accuracy_results.append(result)
        
        novelty_results = self.evaluate_novelty_detection(master_df)
        misclass_results = self.analyze_misclassifications(master_df)
        purity_results = self.calculate_cluster_purity(master_df)
        
        # Generate report
        self.generate_report(master_df, accuracy_results, novelty_results, 
                           misclass_results, purity_results)
        
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*80}")


def main():
    evaluator = EnhancedEvaluator(
        predictions_file="dataset/annotation/sequence_predictions.csv",
        ground_truth_file="dataset/ssu_accession_full_lineage.tsv",
        output_dir="dataset/evaluation"
    )
    
    try:
        evaluator.run_evaluation()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())