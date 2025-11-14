import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, Any

def plot_training_loss(output_dir: Path, history_file: str = 'models/training_history_ae.json'):
    """Plots the training and validation loss curve from a history JSON file."""
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Training history file not found at {history_file}. Skipping loss plot.")
        return
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {history_file}. Skipping loss plot.")
        return

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        print("Warning: Training history file is missing loss data. Skipping loss plot.")
        return

    epochs = range(1, len(train_loss) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, color='#2E86AB', label='Training Loss')
    ax.plot(epochs, val_loss, color='#E63946', label='Validation Loss')
    
    ax.set_title('Encoder Training Loss Curve (MSE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'encoder_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'encoder_loss_curve.png'}")


def plot_all_novelty_pies(data: Dict[str, Any], output_dir: Path):
    """Generates the three novelty breakdown pie charts."""
    
    total_sequences_all = data['dataset_overview']['total_sequences']
    gt_sequences = data['dataset_overview']['sequences_with_ground_truth']
    no_gt_sequences = data['dataset_overview']['sequences_without_ground_truth']
    
    # Values extracted from previous analysis (master_evaluation_table.csv analysis)
    # Total noise sequences (cluster_id = -1)
    noise_sequences_count = 35382 
    clustered_sequences_count = total_sequences_all - noise_sequences_count
    
    # Total sequences classified as novel (noise + novel_cluster)
    predicted_novel_count = 44125 
    predicted_known_count = total_sequences_all - predicted_novel_count

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    colors = plt.cm.Set2.colors
    explode_novel = [0, 0.1] 

    # Plot 1: Total Sequences Distribution (Ground Truth Breakdown)
    total_pie_data = [gt_sequences, no_gt_sequences]
    total_pie_labels = [f'With GT (Known): {gt_sequences:,}', f'Without GT (Novel): {no_gt_sequences:,}']
    axes[0].pie(total_pie_data, labels=total_pie_labels, autopct='%1.1f%%', startangle=90, 
                colors=[colors[2], colors[3]], explode=explode_novel)
    axes[0].set_title('Total Sequences: Ground Truth Breakdown', fontsize=14, fontweight='bold')

    # Plot 2: Clustering Noise Breakdown
    clustering_pie_data = [clustered_sequences_count, noise_sequences_count]
    clustering_pie_labels = [f'Clustered (Assigned ID): {clustered_sequences_count:,}', 
                            f'Noise (cluster_id = -1): {noise_sequences_count:,}']
    axes[1].pie(clustering_pie_data, labels=clustering_pie_labels, autopct='%1.1f%%', startangle=90,
                colors=[colors[4], colors[5]], explode=explode_novel)
    axes[1].set_title('Predicted Novelty by Clustering (Noise)', fontsize=14, fontweight='bold')

    # Plot 3: Annotation Prediction Breakdown
    annotation_pie_data = [predicted_known_count, predicted_novel_count]
    annotation_pie_labels = [f'Predicted Known: {predicted_known_count:,}', 
                             f'Predicted Novel (Cluster/Noise): {predicted_novel_count:,}']
    axes[2].pie(annotation_pie_data, labels=annotation_pie_labels, autopct='%1.1f%%', startangle=90,
                colors=[colors[0], colors[1]], explode=explode_novel)
    axes[2].set_title('Total Predicted Novel Sequences', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'novelty_breakdown_pies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'novelty_breakdown_pies.png'}")


def generate_evaluation_plots(json_file: str = 'dataset/evaluation/evaluation_metrics.json', output_dir_name: str = 'results/plots'):
    """
    Generates and saves a suite of evaluation plots for taxonomic classification 
    and novelty detection.
    """
    
    # --- Configuration and Setup ---
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Load Data and Extract Metrics ---
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Evaluation metrics file not found at {json_file}")
        return

    # 1. Accuracy Metrics
    taxonomic_levels = [item['level'].capitalize() for item in data['accuracy_by_level']]
    accuracies = [item['accuracy'] * 100 for item in data['accuracy_by_level']]
    correct_seqs = [item['correct'] for item in data['accuracy_by_level']]
    incorrect_seqs = [item['incorrect'] for item in data['accuracy_by_level']]

    # 2. Novelty Detection Metrics
    nov = data['novelty_detection']
    confusion_data = np.array([[nov['true_positives'], nov['false_negatives']], 
                               [nov['false_positives'], nov['true_negatives']]])
    novelty_metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    novelty_values = [nov['precision'], nov['recall'], nov['f1_score'], nov['accuracy']]

    # 3. Cluster Purity
    cp = data['cluster_purity_genus']
    purity_categories = ['High\n(>0.9)', 'Medium\n(0.7-0.9)', 'Low\n(<0.7)']
    purity_counts = [cp['high_purity_clusters'], cp['medium_purity_clusters'], cp['low_purity_clusters']]
    total_clusters = cp['total_clusters']
    mean_purity = cp['mean_purity'] # For dashboard

    # 4. Misclassifications
    misclass = data['misclassifications_genus']['top_patterns'][:10] # Top 10
    misclass_from = [item['ground_truth'] for item in misclass]
    misclass_to = [item['predicted'] for item in misclass]
    misclass_counts = [item['count'] for item in misclass]

    # 5. Data Sources 
    source_names = [Path(src).stem.replace('_accession_full_lineage', '') for src in data['ground_truth_sources']]
    source_counts = [65678, 21278, 7951, 6574] # Hardcoded counts
    
    # Define colors for reuse
    purity_colors = ['#06A77D', '#F4A261', '#E63946']
    colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    novelty_bar_colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D']

    # =========================================================================
    # PLOTTING THE 8 ORIGINAL CHARTS
    # =========================================================================
    
    # 1. Classification Accuracy Cascade
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(taxonomic_levels, accuracies, marker='o', linewidth=3, markersize=10, color='#2E86AB')
    ax.fill_between(range(len(taxonomic_levels)), accuracies, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Taxonomic Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy Cascade by Taxonomic Level', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    for i, (level, acc) in enumerate(zip(taxonomic_levels, accuracies)):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_cascade.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correct vs Incorrect Counts
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(taxonomic_levels))
    width = 0.35
    ax.bar(x - width/2, correct_seqs, width, label='Correct', color='#06A77D')
    ax.bar(x + width/2, incorrect_seqs, width, label='Incorrect', color='#D5573B')
    ax.set_xlabel('Taxonomic Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax.set_title('Correct vs Incorrect Classifications by Taxonomic Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(taxonomic_levels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'correct_vs_incorrect.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Novelty Detection Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Predicted Novel', 'Predicted Known'],
                yticklabels=['Actually Novel', 'Actually Known'],
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='black')
    ax.set_title('Novelty Detection Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'novelty_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Novelty Detection Performance Metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(novelty_metrics, novelty_values, color=novelty_bar_colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Novelty Detection Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, novelty_values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'novelty_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Cluster Purity Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(purity_categories, purity_counts, color=purity_colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Purity Distribution (Genus Level)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, purity_counts):
        height = bar.get_height()
        percentage = (count / total_clusters) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_purity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Top Misclassification Patterns
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f'{f} → {t}' for f, t in zip(misclass_from, misclass_to)]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, misclass_counts, color='#E63946', edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    ax.set_title('Top Misclassification Patterns (Genus Level)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (bar, count) in enumerate(zip(bars, misclass_counts)):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_misclassifications.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Ground Truth Data Sources Distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    source_labels_pie = [f'{name}\n({count:,})' for name, count in zip(source_names, source_counts)]
    ax.pie(source_counts, labels=source_labels_pie, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Ground Truth Data Sources Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_sources.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Summary Dashboard (Consolidated Plot)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(taxonomic_levels, accuracies, marker='o', linewidth=3, markersize=8, color='#2E86AB')
    ax1.fill_between(range(len(taxonomic_levels)), accuracies, alpha=0.3, color='#2E86AB')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Cascade', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Novel', 'Known'], yticklabels=['Novel', 'Known'],
                cbar=False, ax=ax2, linewidths=1)
    ax2.set_title('Novelty Detection', fontweight='bold', fontsize=12)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(3), purity_counts, color=purity_colors, edgecolor='black')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['High', 'Med', 'Low'], fontsize=9)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Cluster Purity', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.barh(novelty_metrics, novelty_values, color=novelty_bar_colors)
    ax4.set_xlim(0, 1)
    ax4.set_title('Novelty Metrics', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.tick_params(axis='y', labelsize=9)

    ax5 = fig.add_subplot(gs[2, 0])
    short_names = [name.split('_')[0] for name in source_names]
    ax5.pie(source_counts, labels=short_names, autopct='%1.0f%%',
            colors=colors_pie, textprops={'fontsize': 9})
    ax5.set_title('Data Sources', fontweight='bold', fontsize=12)

    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    total_seqs_all = data['dataset_overview']['total_sequences']
    with_gt = data['dataset_overview']['sequences_with_ground_truth']
    
    genus_acc = next((item['accuracy'] * 100 for item in data['accuracy_by_level'] if item['level'] == 'genus'), 0.0)
    species_acc = next((item['accuracy'] * 100 for item in data['accuracy_by_level'] if item['level'] == 'species'), 0.0)

    f1 = nov['f1_score']

    key_stats = [
        f"Total Sequences: {total_seqs_all:,}",
        f"With Ground Truth: {with_gt:,} ({with_gt/total_seqs_all*100:.1f}%)",
        f"Total Clusters: {total_clusters:,}",
        f"Genus Accuracy: {genus_acc:.2f}%",
        f"Species Accuracy: {species_acc:.2f}%",
        f"Mean Cluster Purity: {mean_purity:.3f}",
        f"Novelty F1-Score: {f1:.3f}",
    ]
    y_start = 0.9
    for i, stat in enumerate(key_stats):
        is_low = ('Accuracy' in stat and float(stat.split(':')[1].replace('%', '').strip()) < 50)
        color = '#E63946' if is_low or ('F1-Score' in stat and float(stat.split(':')[1].strip()) < 0.5) else '#2E86AB'
        
        ax6.text(0.1, y_start - i*0.11, stat, fontsize=11, fontweight='normal', 
                 color=color, transform=ax6.transAxes)
    ax6.set_title('Key Statistics', fontweight='bold', fontsize=12, loc='left')

    fig.suptitle('Taxonomic Classification Evaluation Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # CALLING NEW PLOTTING FUNCTIONS
    # =========================================================================
    plot_all_novelty_pies(data, output_dir)
    plot_training_loss(output_dir)


def main():
    """Main program execution block."""
    generate_evaluation_plots()
    print("\nEvaluation visualization generation complete! Saved plots to results/plots/.")


if __name__ == "__main__":
    main()