#!/usr/bin/env python3
version='0.0.1'
import subprocess
import requests
import os

def check_amd_gpu():
    """Check AMD GPU information"""
    try:
        # Check if AMD GPU is present
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        amd_gpus = [line for line in result.stdout.split('\n') if 'AMD' in line and 'Vega' in line]
        
        print("AMD GPU Detection:")
        for gpu in amd_gpus:
            print(f"  ✅ {gpu.strip()}")
            
        # Check Vulkan
        result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Vulkan is available")
        else:
            print("  ❌ Vulkan not available")
            
    except Exception as e:
        print(f"GPU check error: {e}")

def check_ollama_gpu_usage(host="http://localhost:11434"):
    """Check if Ollama is using GPU"""
    try:
        response = requests.get(f"{host}/api/ps")
        if response.status_code == 200:
            processes = response.json().get('models', [])
            for proc in processes:
                if 'gpu' in proc:
                    gpu_info = proc['gpu']
                    print("Ollama GPU Usage:")
                    print(f"  GPU: {gpu_info.get('name', 'Unknown')}")
                    print(f"  Memory: {gpu_info.get('memory', {})}")
                    return True
        print("  ❌ No GPU information available from Ollama")
        return False
    except Exception as e:
        print(f"Ollama GPU check error: {e}")
        return False

if __name__ == "__main__":
    check_amd_gpu()
    check_ollama_gpu_usage()