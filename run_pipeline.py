"""
run_pipeline.py
===============
Complete Pipeline Execution Script

Runs the full eDNA analysis pipeline with enhanced evaluation

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-steps 1,2  # Skip certain steps
    python run_pipeline.py --only-step 8     # Run only annotation
"""

import subprocess
import logging
from pathlib import Path
import argparse
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Execute complete eDNA analysis pipeline"""
    
    def __init__(self, skip_steps=None, only_step=None):
        self.skip_steps = skip_steps or []
        self.only_step = only_step
        
        self.pipeline_steps = [
            ('fasta_extraction.py', 'Step 1: FASTA Extraction from BLAST Databases'),
            ('minimal_preprocessing.py', 'Step 2: Minimal Preprocessing'),
            ('metadata_extraction.py', 'Step 3: Metadata Extraction'),
            ('cgr_transformation.py', 'Step 4: CGR Transformation'),
            ('cnn_encoder.py', 'Step 5: CNN Encoder Training'),
            ('generate_embeddings.py', 'Step 6: Generate Embeddings'),
            ('clustering.py', 'Step 7: HDBSCAN Clustering'),
            ('cluster_annotation.py', 'Step 8: Cluster Annotation via BLAST'),
            ('evaluation.py', 'Step 9: Enhanced Evaluation'),
            ('visualization.py', 'Step 10: Visualization'),
        ]
    
    def run_script(self, script_name: str, description: str, step_num: int) -> bool:
        """Run a Python script and return success status"""
        
        # Check if this step should be skipped
        if step_num in self.skip_steps:
            logger.info(f"\nSKIPPING: {description}")
            return True
        
        # If only_step is set, skip all other steps
        if self.only_step and step_num != self.only_step:
            return True
        
        logger.info(f"\n{'='*80}")
        logger.info(f"{description}")
        logger.info(f"{'='*80}\n")
        
        # Check if script exists
        if not Path(script_name).exists():
            logger.error(f"Script not found: {script_name}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, script_name],
                check=True,
                capture_output=False
            )
            logger.info(f"\n{description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n{description} failed with error code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"\n{description} failed: {e}")
            return False
    
    def run(self):
        """Execute complete pipeline"""
        
        logger.info("\n" + "="*80)
        logger.info("EDNA ANALYSIS PIPELINE - COMPLETE EXECUTION")
        logger.info("="*80 + "\n")
        
        if self.skip_steps:
            logger.info(f"Skipping steps: {self.skip_steps}")
        if self.only_step:
            logger.info(f"Running only step: {self.only_step}")
        
        results = {}
        
        for step_num, (script, description) in enumerate(self.pipeline_steps, 1):
            success = self.run_script(script, description, step_num)
            results[script] = success
            
            if not success and self.only_step is None:
                logger.error(f"\nPipeline halted due to failure in: {description}")
                break
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80 + "\n")
        
        for script, success in results.items():
            if self.only_step:
                if int(script.split()[0].replace('Step', '').rstrip(':')) == self.only_step:
                    status = "SUCCESS" if success else "FAILED"
                    logger.info(f"{script:40s} : {status}")
            else:
                status = "SUCCESS" if success else "FAILED"
                logger.info(f"{script:40s} : {status}")
        
        if all(results.values()):
            logger.info("\nAll pipeline steps completed successfully")
            return 0
        else:
            logger.error("\nSome pipeline steps failed")
            return 1


def main():
    parser = argparse.ArgumentParser(description='Run complete eDNA analysis pipeline')
    parser.add_argument('--skip-steps', type=str, help='Comma-separated list of step numbers to skip (e.g., 1,2,3)')
    parser.add_argument('--only-step', type=int, help='Run only this step number')
    
    args = parser.parse_args()
    
    skip_steps = []
    if args.skip_steps:
        skip_steps = [int(x.strip()) for x in args.skip_steps.split(',')]
    
    runner = PipelineRunner(skip_steps=skip_steps, only_step=args.only_step)
    
    return runner.run()


if __name__ == "__main__":
    exit(main())


# =============================================================================
# QUICK START GUIDE
# =============================================================================
"""
COMPLETE PIPELINE (All Steps):
    python run_pipeline.py

SKIP DATA PREPARATION (if already done):
    python run_pipeline.py --skip-steps 1,2,3

RUN ONLY ANNOTATION:
    python run_pipeline.py --only-step 8

RUN ONLY EVALUATION:
    python run_pipeline.py --only-step 9

RUN ANNOTATION + EVALUATION:
    python run_pipeline.py --skip-steps 1,2,3,4,5,6,7

INDIVIDUAL SCRIPTS:
    python cluster_annotation.py
    python cluster_annotation.py --local-blast
    python cluster_annotation.py --batch-size 25 --resume
    python evaluation.py
"""