#!/usr/bin/env python3
import requests
import json
import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import sys

class OllamaBenchmark:
    def __init__(self, host="http://localhost:11434", cpu_host="http://localhost:11435"):
        self.host = host
        self.cpu_host = cpu_host
        self.results = []
        
    def check_ollama_health(self, host=None):
        """Check if Ollama is running and get available models"""
        target_host = host or self.host
        try:
            response = requests.get(f"{target_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"Available models on {target_host}:")
                for model in models:
                    print(f"  - {model['name']} (size: {model.get('size', 'N/A')})")
                return True, models
            return False, []
        except Exception as e:
            print(f"Error connecting to {target_host}: {e}")
            return False, []
    
    def get_system_info(self):
        """Get system and GPU information"""
        try:
            # Get Ollama system info
            response = requests.get(f"{self.host}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                print(f"Ollama Version: {version_info.get('version')}")
            
            # Try to get GPU info through ollama
            response = requests.get(f"{self.host}/api/ps")
            if response.status_code == 200:
                processes = response.json().get('models', [])
                for proc in processes:
                    if 'gpu' in proc:
                        print(f"GPU Utilization: {proc.get('gpu', {})}")
            
        except Exception as e:
            print(f"Could not get system info: {e}")
    
    def run_inference(self, host, model_name, prompt, use_gpu=True):
        """Run single inference and return timing"""
        start_time = time.time()
        
        payload = {
            'model': model_name,
            'prompt': prompt,
            'stream': False,
            'options': {}
        }
        
        # Only set GPU options if we're controlling it
        if host == self.cpu_host:
            payload['options']['num_gpu'] = 0
        elif not use_gpu:
            payload['options']['num_gpu'] = 0
        
        try:
            response = requests.post(
                f'{host}/api/generate',
                json=payload,
                timeout=300
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                tokens = len(data.get('response', '').split())
                
                return {
                    'time': end_time - start_time,
                    'tokens': tokens,
                    'response': data.get('response', ''),
                    'total_duration': data.get('total_duration', 0),
                    'load_duration': data.get('load_duration', 0),
                    'prompt_eval_duration': data.get('prompt_eval_duration', 0),
                    'eval_duration': data.get('eval_duration', 0),
                    'eval_count': data.get('eval_count', 0)
                }
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timeout")
            return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def benchmark_endpoint(self, host, model_name, prompts, iterations=3, mode="GPU"):
        """Run benchmark on specific endpoint"""
        print(f"\n=== Running {mode} Benchmark ===")
        print(f"Endpoint: {host}")
        print(f"Model: {model_name}")
        print(f"Iterations: {iterations}")
        
        times = []
        tokens_per_second = []
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}:")
            
            for j, prompt in enumerate(prompts):
                print(f"  Prompt {j+1}/{len(prompts)}...", end=" ")
                result = self.run_inference(host, model_name, prompt, use_gpu=(mode=="GPU"))
                
                if result:
                    times.append(result['time'])
                    if result['eval_duration'] > 0:
                        tps = result['eval_count'] / (result['eval_duration'] / 1e9)
                        tokens_per_second.append(tps)
                    
                    self.results.append({
                        'timestamp': datetime.now().isoformat(),
                        'model': model_name,
                        'mode': mode,
                        'endpoint': host,
                        'iteration': i + 1,
                        'prompt_id': j + 1,
                        'prompt_length': len(prompt),
                        'total_time': result['time'],
                        'tokens_generated': result['tokens'],
                        'tokens_per_second': tps if result['eval_duration'] > 0 else 0,
                        'total_duration_ns': result['total_duration'],
                        'eval_duration_ns': result['eval_duration'],
                        'load_duration_ns': result['load_duration'],
                        'prompt_eval_duration_ns': result['prompt_eval_duration'],
                        'eval_count': result['eval_count']
                    })
                    print(f"✓ ({result['time']:.2f}s, {tps:.1f} tok/s)")
                else:
                    print("✗ Failed")
                    return None
        
        if times:
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            avg_tps = statistics.mean(tokens_per_second) if tokens_per_second else 0
            
            print(f"\n{mode} Results:")
            print(f"  Average time: {avg_time:.2f}s (±{std_time:.2f}s)")
            print(f"  Average tokens/second: {avg_tps:.2f}")
            
            return {
                'mode': mode,
                'model': model_name,
                'endpoint': host,
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_tps': avg_tps,
                'iterations': iterations
            }
        
        return None
    
    def compare_performance(self, model_name, prompts, iterations=3):
        """Compare GPU (Vulkan) vs CPU performance"""
        print("=== Performance Comparison ===")
        
        # Check main Ollama instance (GPU/Vulkan)
        gpu_healthy, models = self.check_ollama_health(self.host)
        if not gpu_healthy:
            print(f"❌ Main Ollama instance not available at {self.host}")
            return None
        
        # Check if model is available
        model_available = any(model_name in model['name'] for model in models)
        if not model_available:
            print(f"❌ Model {model_name} not found in main instance")
            print("Available models:", [m['name'] for m in models])
            return None
        
        # Get system info
        self.get_system_info()
        
        # Benchmark main instance (GPU/Vulkan)
        gpu_results = self.benchmark_endpoint(
            self.host, model_name, prompts, iterations, "GPU"
        )
        
        # Benchmark CPU instance (if available)
        cpu_results = None
        cpu_healthy, _ = self.check_ollama_health(self.cpu_host)
        if cpu_healthy:
            cpu_results = self.benchmark_endpoint(
                self.cpu_host, model_name, prompts, iterations, "CPU"
            )
        else:
            print(f"⚠️ CPU comparison instance not available at {self.cpu_host}")
            print("Running GPU-only benchmark")
        
        # Compare results
        if gpu_results and cpu_results:
            speedup = cpu_results['avg_time'] / gpu_results['avg_time']
            tps_improvement = gpu_results['avg_tps'] / cpu_results['avg_tps'] if cpu_results['avg_tps'] > 0 else 0
            
            print(f"\n{'='*50}")
            print(f"🎯 PERFORMANCE COMPARISON RESULTS")
            print(f"{'='*50}")
            print(f"GPU vs CPU Speedup: {speedup:.2f}x")
            print(f"Tokens/second improvement: {tps_improvement:.2f}x")
            print(f"\nGPU Performance:")
            print(f"  Average time: {gpu_results['avg_time']:.2f}s")
            print(f"  Tokens/sec: {gpu_results['avg_tps']:.2f}")
            print(f"\nCPU Performance:")
            print(f"  Average time: {cpu_results['avg_time']:.2f}s")
            print(f"  Tokens/sec: {cpu_results['avg_tps']:.2f}")
            print(f"{'='*50}")
            
            return {
                'speedup': speedup,
                'tps_improvement': tps_improvement,
                'gpu': gpu_results,
                'cpu': cpu_results
            }
        elif gpu_results:
            print(f"\nGPU-only Results:")
            print(f"  Average time: {gpu_results['avg_time']:.2f}s")
            print(f"  Tokens/sec: {gpu_results['avg_tps']:.2f}")
            return {'gpu': gpu_results}
        
        return None
    
    def generate_report(self):
        """Generate comprehensive report and visualizations"""
        if not self.results:
            print("No results to report")
            return
        
        # Create results directory
        os.makedirs('/app/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results
        df = pd.DataFrame(self.results)
        csv_path = f'/app/results/benchmark_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved to: {csv_path}")
        
        # Generate summary
        summary_path = f'/app/results/summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_runs': len(self.results),
                'models_tested': list(df['model'].unique()),
                'modes_tested': list(df['mode'].unique())
            }, f, indent=2)
        
        # Generate visualizations if we have both GPU and CPU data
        if 'GPU' in df['mode'].values and 'CPU' in df['mode'].values:
            self._generate_plots(df, timestamp)
    
    def _generate_plots(self, df, timestamp):
        """Generate comparison plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time comparison
        sns.boxplot(data=df, x='mode', y='total_time', ax=axes[0,0])
        axes[0,0].set_title('Inference Time Comparison')
        axes[0,0].set_ylabel('Time (seconds)')
        
        # Tokens per second comparison
        sns.boxplot(data=df, x='mode', y='tokens_per_second', ax=axes[0,1])
        axes[0,1].set_title('Tokens per Second Comparison')
        axes[0,1].set_ylabel('Tokens/Second')
        
        # Time by iteration
        sns.lineplot(data=df, x='iteration', y='total_time', hue='mode', ax=axes[1,0])
        axes[1,0].set_title('Inference Time by Iteration')
        axes[1,0].set_ylabel('Time (seconds)')
        
        # Tokens per second by iteration
        sns.lineplot(data=df, x='iteration', y='tokens_per_second', hue='mode', ax=axes[1,1])
        axes[1,1].set_title('Tokens per Second by Iteration')
        axes[1,1].set_ylabel('Tokens/Second')
        
        plt.tight_layout()
        plot_path = f'/app/results/comparison_plot_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")
        plt.close()

def main():
    # Configuration from environment
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    cpu_host = os.getenv('OLLAMA_CPU_HOST', 'http://localhost:11435')
    model_name = os.getenv('BENCHMARK_MODEL', 'llama2:7b')
    iterations = int(os.getenv('BENCHMARK_ITERATIONS', '3'))
    
    # Test prompts
    test_prompts = [
        "Explain the concept of machine learning in 2-3 sentences.",
        "What are the main advantages of renewable energy sources?",
        "Describe the process of photosynthesis briefly.",
        "Write a short haiku about technology and nature.",
        "What is the difference between AI and traditional programming?"
    ]
    
    print("🚀 Starting Ollama Benchmark")
    print(f"Target Ollama: {host}")
    print(f"CPU Comparison: {cpu_host}")
    print(f"Model: {model_name}")
    print(f"Iterations: {iterations}")
    
    benchmark = OllamaBenchmark(host=host, cpu_host=cpu_host)
    
    # Run comparison
    comparison = benchmark.compare_performance(
        model_name=model_name,
        prompts=test_prompts,
        iterations=iterations
    )
    
    # Generate report
    benchmark.generate_report()
    
    if comparison:
        print("\n✅ Benchmark completed successfully!")
        if 'speedup' in comparison:
            print(f"🎯 Final Result: GPU is {comparison['speedup']:.2f}x faster than CPU")
    else:
        print("\n❌ Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()