#!/usr/bin/env python3
"""
Custom 4DBInfer CLI Interface for hHGTN Integration
===================================================

This provides a working CLI interface for the completed Stage 11 implementation,
bypassing DGL compatibility issues while maintaining the 4DBInfer workflow.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add our adapter to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_cli():
    """Setup command line interface"""
    parser = argparse.ArgumentParser(
        description="4DBInfer CLI for hHGTN Integration (Stage 11 Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-solutions          # List available solutions
  %(prog)s --run-solution hhgtn      # Run hHGTN solution  
  %(prog)s --run-ablation            # Run ablation study
  %(prog)s --validate-integration    # Validate integration
  %(prog)s --benchmark               # Run full benchmark
        """
    )
    
    parser.add_argument('--list-solutions', action='store_true',
                       help='List all available GML solutions')
    
    parser.add_argument('--run-solution', type=str, metavar='NAME',
                       help='Run specific solution (e.g., hhgtn)')
    
    parser.add_argument('--run-ablation', action='store_true', 
                       help='Run ablation study with all configurations')
    
    parser.add_argument('--validate-integration', action='store_true',
                       help='Validate 4DBInfer integration')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Run complete benchmarking suite')
    
    parser.add_argument('--config', type=str, metavar='PATH',
                       help='Path to configuration file')
    
    parser.add_argument('--output-dir', type=str, metavar='PATH', 
                       default='./dbinfer_outputs',
                       help='Output directory for results')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser

def list_solutions():
    """List available GML solutions"""
    print("4DBInfer - Available GML Solutions")
    print("=" * 40)
    print()
    print("✅ hhgtn        - Hypergraph Heterogeneous Graph Transformer Network")
    print("                 Status: Fully integrated (Stage 11 complete)")
    print("                 Ablation controls: SpotTarget, CUSP, TRD, Memory")
    print()
    print("ℹ️  Other solutions would be listed here if available")
    print()

def run_solution(solution_name, output_dir, verbose=False):
    """Run a specific solution"""
    if solution_name.lower() == 'hhgtn':
        print(f"Running hHGTN solution...")
        print(f"Output directory: {output_dir}")
        
        # Import and run our integration evaluation
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 'run_integration_eval.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ hHGTN solution completed successfully")
                if verbose:
                    print(result.stdout)
            else:
                print("❌ hHGTN solution failed")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running hHGTN solution: {e}")
    else:
        print(f"❌ Unknown solution: {solution_name}")
        print("Available solutions: hhgtn")

def run_ablation_study(output_dir, verbose=False):
    """Run ablation study"""
    print("Running ablation study...")
    print(f"Output directory: {output_dir}")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'run_ablation_studies.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ablation study completed successfully")
            if verbose:
                print(result.stdout)
        else:
            print("❌ Ablation study failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running ablation study: {e}")

def validate_integration(verbose=False):
    """Validate 4DBInfer integration"""
    print("Validating 4DBInfer integration...")
    
    try:
        # Run adapter tests
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/test_4dbinfer_adapter.py', '-v' if verbose else '-q'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Integration validation passed")
            if verbose:
                print(result.stdout)
        else:
            print("❌ Integration validation failed")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error validating integration: {e}")

def run_benchmark(output_dir, verbose=False):
    """Run complete benchmark suite"""
    print("Running complete benchmark suite...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Benchmark Suite - Stage 11 Implementation")
    print("=" * 50)
    
    # Phase validation
    phases = [
        ("Phase A: Baseline Verification", "run_baseline_smoke_v2.py"),
        ("Phase C: Adapter Tests", "tests/test_4dbinfer_adapter.py"),
        ("Phase D: Integration Evaluation", "run_integration_eval.py"),  
        ("Phase E: Ablation Studies", "run_ablation_studies.py")
    ]
    
    results = {}
    
    for phase_name, script in phases:
        print(f"\n{phase_name}")
        print("-" * 30)
        
        try:
            if script.endswith('.py') and not script.startswith('test_') and 'test_' not in script:
                result = subprocess.run([sys.executable, script], 
                                      capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run([sys.executable, '-m', 'pytest', script, '-v'], 
                                      capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {phase_name} - PASSED")
                results[phase_name] = "PASSED"
            else:
                print(f"❌ {phase_name} - FAILED")
                results[phase_name] = "FAILED"
                if verbose:
                    print(result.stderr)
                    
        except subprocess.TimeoutExpired:
            print(f"⏱️ {phase_name} - TIMEOUT")
            results[phase_name] = "TIMEOUT"
        except Exception as e:
            print(f"❌ {phase_name} - ERROR: {e}")
            results[phase_name] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for phase, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"{status_icon} {phase}: {result}")
    
    print(f"\nOverall: {passed}/{total} phases passed ({passed/total*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    
    benchmark_data = {
        "timestamp": timestamp,
        "phase_results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": f"{passed/total*100:.1f}%"
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")

def main():
    """Main CLI entry point"""
    parser = setup_cli()
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print("4DBInfer CLI - hHGTN Integration (Stage 11)")
    print("=" * 45)
    print()
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Execute commands
    if args.list_solutions:
        list_solutions()
    
    elif args.run_solution:
        run_solution(args.run_solution, args.output_dir, args.verbose)
    
    elif args.run_ablation:
        run_ablation_study(args.output_dir, args.verbose)
    
    elif args.validate_integration:
        validate_integration(args.verbose)
    
    elif args.benchmark:
        run_benchmark(args.output_dir, args.verbose)
    
    else:
        print("No valid command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()
