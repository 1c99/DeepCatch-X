#!/usr/bin/env python3
"""
Optimized Ray-based DCX Inference System with Module-Specific Workers
Each module has a dedicated worker that loads the model once
"""
import os
import sys
import time
import argparse
import glob
import ray
import numpy as np
from datetime import datetime

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))


@ray.remote
class ModuleSpecificWorker:
    """Base class for module-specific workers"""
    
    def __init__(self, module_name, config_path, device='auto'):
        """Initialize worker with specific module"""
        import torch
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Import and initialize the inference model once
        from inference import UnifiedDCXInference
        
        self.module_name = module_name
        self.inference = UnifiedDCXInference(
            config_path, 
            device_override=self.device,
            module_name=module_name,
            batch_mode=True
        )
        
        print(f"Initialized {module_name} worker on {self.device}")
    
    def process_file(self, input_file, output_dir, output_format='png', 
                    output_size='original', no_calculations=False, 
                    collect_measurements=False, lung_mask_path=None):
        """Process a single file with this module"""
        try:
            # Create output filename
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Determine output filename based on module
            if self.module_name == 'bone_supp':
                output_filename = f"{input_basename}_{self.module_name}.{output_format}"
            elif self.module_name == 'aorta0':
                output_filename = f"{input_basename}_aorta_asc.{output_format}"
            elif self.module_name == 'aorta1':
                output_filename = f"{input_basename}_aorta_desc.{output_format}"
            else:
                output_filename = f"{input_basename}_{self.module_name}.{output_format}"
                
            output_file = os.path.join(output_dir, output_filename)
            
            # Update settings for this file
            self.inference.output_format = output_format
            self.inference.output_size = output_size
            self.inference.unified_csv = collect_measurements
            
            # Configure calculations
            if no_calculations:
                self.inference.config['calculate_area'] = False
                self.inference.config['calculate_volume'] = False
            else:
                volume_modules = ['heart', 'lung', 'heart_volumetry', 'lung_volumetry', 't12l1_regression']
                if self.module_name not in volume_modules:
                    self.inference.config['calculate_volume'] = False
            
            # Process the file
            results = self.inference.process(input_file, output_file, lung_mask_path)
            
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'success',
                'output_file': output_file,
                'results': results
            }
            
        except Exception as e:
            import traceback
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


@ray.remote
class LungDependentWorker:
    """Worker for modules that depend on lung mask"""
    
    def __init__(self, module_name, config_path, device='auto'):
        """Initialize worker with specific module"""
        import torch
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Import and initialize the inference model once
        from inference import UnifiedDCXInference
        
        self.module_name = module_name
        self.inference = UnifiedDCXInference(
            config_path, 
            device_override=self.device,
            module_name=module_name,
            batch_mode=True
        )
        
        print(f"Initialized {module_name} worker (lung-dependent) on {self.device}")
    
    def process_file_with_lung(self, input_file, output_dir, lung_results,
                             output_format='png', output_size='original', 
                             no_calculations=False, collect_measurements=False):
        """Process file with lung dependency check"""
        
        # Check if lung was successful
        if lung_results.get('status') != 'success':
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'skipped',
                'error': 'Lung mask required but not available'
            }
        
        # Find lung mask path
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        lung_mask_path = None
        
        # First try temp file
        temp_lung_path = os.path.join(output_dir, f'{input_basename}_lung_temp2048.nii')
        if os.path.exists(temp_lung_path):
            lung_mask_path = temp_lung_path
        else:
            # Try other formats
            potential_files = [
                os.path.join(output_dir, f'{input_basename}_lung.{output_format}'),
                os.path.join(output_dir, f'{input_basename}_lung.nii'),
                os.path.join(output_dir, f'{input_basename}_lung.png')
            ]
            for f in potential_files:
                if os.path.exists(f):
                    lung_mask_path = f
                    break
        
        if not lung_mask_path:
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'skipped',
                'error': 'Lung mask file not found'
            }
        
        # Process with lung mask
        try:
            # Create output filename
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            output_filename = f"{input_basename}_{self.module_name}.{output_format}"
            output_file = os.path.join(output_dir, output_filename)
            
            # Update settings
            self.inference.output_format = output_format
            self.inference.output_size = output_size
            self.inference.unified_csv = collect_measurements
            
            # Configure calculations
            if no_calculations:
                self.inference.config['calculate_area'] = False
                self.inference.config['calculate_volume'] = False
            
            # Process the file with lung mask
            results = self.inference.process(input_file, output_file, lung_mask_path)
            
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'success',
                'output_file': output_file,
                'results': results
            }
            
        except Exception as e:
            import traceback
            return {
                'file': input_file,
                'module': self.module_name,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


def create_module_workers(modules_to_run, device='auto', num_gpus=0):
    """Create dedicated workers for each module"""
    workers = {}
    
    # Modules that need lung mask
    lung_dependent = ['covid', 'vessel', 'tb']
    
    for module in modules_to_run:
        if module in ['ctr', 'peripheral', 'diameter']:
            # Skip post-processing modules for now
            continue
            
        # Determine config path
        base_module = 'aorta' if module in ['aorta0', 'aorta1'] else module
        config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{base_module}.yaml')
        
        if not os.path.exists(config_path):
            print(f"Warning: Config not found for {module}: {config_path}")
            continue
        
        # Create appropriate worker type
        if module in lung_dependent:
            workers[module] = LungDependentWorker.options(
                num_gpus=num_gpus
            ).remote(module, config_path, device)
        else:
            workers[module] = ModuleSpecificWorker.options(
                num_gpus=num_gpus
            ).remote(module, config_path, device)
    
    return workers


def main():
    parser = argparse.ArgumentParser(description='Optimized Ray-based DCX Inference')
    
    # Module selection
    parser.add_argument('--module', type=str, action='append',
                       choices=['lung', 'heart', 'airway', 'bone', 'aorta', 't12l1', 'laa',
                               'tb', 'bone_supp', 'covid', 'vessel', 'aorta0', 'aorta1',
                               'ctr', 'peripheral', 'diameter'],
                       help='Module to use (can be specified multiple times)')
    parser.add_argument('--all_modules', action='store_true',
                       help='Run all modules')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input DICOM file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    
    # Ray options
    parser.add_argument('--num_gpus_per_worker', type=float, default=None,
                       help='GPUs per worker (auto-determined if not specified)')
    
    # Processing options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--output_format', type=str, default='png',
                       choices=['nii', 'dcm', 'png'],
                       help='Output format')
    parser.add_argument('--output_size', type=str, default='original',
                       choices=['original', '512', '2048'],
                       help='Output size')
    parser.add_argument('--no_calculations', action='store_true',
                       help='Skip area/volume calculations')
    parser.add_argument('--collect_measurements', action='store_true',
                       help='Collect measurements into CSV')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_modules and not args.module:
        parser.error("Either --module or --all_modules must be specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input files
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = []
        for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
        input_files = sorted(input_files)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} DICOM file(s)")
    
    # Determine modules
    if args.all_modules:
        modules_to_run = ['lung', 'heart', 'airway', 'bone', 'aorta', 't12l1', 'laa',
                         'tb', 'bone_supp', 'covid', 'vessel']
    else:
        modules_to_run = args.module
    
    print(f"Modules to run: {', '.join(modules_to_run)}")
    
    # Initialize Ray
    import torch
    total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.num_gpus_per_worker is None:
        # Auto-determine: share GPUs among module workers
        if total_gpus > 0:
            num_unique_modules = len(set(modules_to_run))
            args.num_gpus_per_worker = max(0.1, total_gpus / num_unique_modules)
        else:
            args.num_gpus_per_worker = 0
    
    print(f"\nInitializing Ray with module-specific workers")
    print(f"GPUs per worker: {args.num_gpus_per_worker}")
    
    ray.init(num_gpus=total_gpus)
    
    # Create module-specific workers
    print("\nCreating dedicated workers for each module...")
    workers = create_module_workers(modules_to_run, args.device, args.num_gpus_per_worker)
    print(f"Created {len(workers)} module workers")
    
    # Process files
    start_time = time.time()
    all_tasks = []
    task_info = []
    
    # Phase 1: Submit all independent modules for all files
    print("\nSubmitting tasks...")
    lung_dependent = ['covid', 'vessel', 'tb']
    independent_modules = [m for m in modules_to_run if m not in lung_dependent and m in workers]
    
    # Store lung results for dependent modules
    lung_results_by_file = {}
    
    for input_file in input_files:
        # Process independent modules
        for module in independent_modules:
            worker = workers[module]
            task = worker.process_file.remote(
                input_file, args.output_dir, args.output_format,
                args.output_size, args.no_calculations, args.collect_measurements
            )
            all_tasks.append(task)
            task_info.append((input_file, module))
            
            # Track lung tasks specially
            if module == 'lung':
                lung_results_by_file[input_file] = task
    
    # Wait for lung results if we have dependent modules
    dependent_modules = [m for m in modules_to_run if m in lung_dependent and m in workers]
    if dependent_modules and lung_results_by_file:
        print("\nWaiting for lung segmentation to complete...")
        lung_results = {}
        for input_file, lung_task in lung_results_by_file.items():
            result = ray.get(lung_task)
            lung_results[input_file] = result
            print(f"  Lung: {os.path.basename(input_file)} - {result['status']}")
        
        # Submit dependent modules
        print("\nSubmitting lung-dependent modules...")
        for input_file in input_files:
            if input_file in lung_results:
                for module in dependent_modules:
                    worker = workers[module]
                    task = worker.process_file_with_lung.remote(
                        input_file, args.output_dir, lung_results[input_file],
                        args.output_format, args.output_size, 
                        args.no_calculations, args.collect_measurements
                    )
                    all_tasks.append(task)
                    task_info.append((input_file, module))
    
    # Process results
    print(f"\nProcessing {len(all_tasks)} tasks...")
    results = []
    
    # Wait for all tasks
    remaining = all_tasks
    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1)
        result = ray.get(ready[0])
        results.append(result)
        
        # Find task info
        task_idx = all_tasks.index(ready[0])
        file, module = task_info[task_idx]
        
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"  {status_symbol} {os.path.basename(file)} - {module} ({result['status']})")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Files: {len(input_files)}, Modules: {len(modules_to_run)}")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Average time per file: {total_time/len(input_files):.1f}s")
    
    # Cleanup
    ray.shutdown()


if __name__ == '__main__':
    main()