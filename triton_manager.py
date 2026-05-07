#!/usr/bin/env python3
"""
Triton Inference Server Manager for Analog Gauge Models

This script helps manage and run Triton Inference Server with the analog gauge models.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class TritonManager:
    def __init__(self, models_dir: str = "models/triton_models", config_dir: str = "models/triton_models"):
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.triton_process = None

    def check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def check_triton_image(self) -> bool:
        """Check if Triton image is available"""
        try:
            result = subprocess.run(["docker", "images", "nvcr.io/nvidia/tritonserver"],
                                  capture_output=True, text=True)
            return "tritonserver" in result.stdout
        except:
            return False

    def pull_triton_image(self):
        """Pull Triton server image"""
        print("Pulling Triton Inference Server image...")
        subprocess.run([
            "docker", "pull", "nvcr.io/nvidia/tritonserver:24.01-py3"
        ], check=True)
        print("Triton image pulled successfully.")

    def start_server(self, http_port: int = 8000, grpc_port: int = 8001, metrics_port: int = 8002):
        """Start Triton server"""
        if not self.check_docker():
            print("Error: Docker is not installed or not running.")
            return False

        if not self.check_triton_image():
            print("Triton image not found. Pulling...")
            self.pull_triton_image()

        if not self.models_dir.exists():
            print(f"Error: Models directory {self.models_dir} does not exist.")
            return False

        # Stop existing server if running
        self.stop_server()

        print(f"Starting Triton server with models from {self.models_dir}")
        print(f"HTTP port: {http_port}, GRPC port: {grpc_port}")

        cmd = [
            "docker", "run", "--rm",
            "-p", f"{http_port}:8000",
            "-p", f"{grpc_port}:8001",
            "-p", f"{metrics_port}:8002",
            "-v", f"{self.models_dir.absolute()}:/models",
            "nvcr.io/nvidia/tritonserver:24.01-py3",
            "tritonserver",
            "--model-repository=/models",
            "--http-port=8000",
            "--grpc-port=8001",
            "--metrics-port=8002",
            "--log-verbose=1"
        ]

        try:
            self.triton_process = subprocess.Popen(cmd)
            print(f"Triton server started with PID: {self.triton_process.pid}")

            # Wait a bit for server to start
            time.sleep(5)

            # Check if server is running
            if self.triton_process.poll() is None:
                print("Triton server is running successfully.")
                return True
            else:
                print("Failed to start Triton server.")
                return False

        except Exception as e:
            print(f"Error starting Triton server: {e}")
            return False

    def stop_server(self):
        """Stop Triton server"""
        if self.triton_process:
            print("Stopping Triton server...")
            self.triton_process.terminate()
            try:
                self.triton_process.wait(timeout=10)
                print("Triton server stopped.")
            except subprocess.TimeoutExpired:
                self.triton_process.kill()
                print("Triton server force killed.")
            self.triton_process = None

        # Also try to stop any running containers
        try:
            result = subprocess.run(["docker", "ps", "-q", "--filter", "ancestor=nvcr.io/nvidia/tritonserver"],
                                  capture_output=True, text=True)
            if result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                for cid in container_ids:
                    subprocess.run(["docker", "stop", cid], capture_output=True)
                print(f"Stopped {len(container_ids)} Triton containers.")
        except:
            pass

    def check_server_health(self, url: str = "localhost:8000") -> bool:
        """Check if Triton server is healthy"""
        try:
            import requests
            response = requests.get(f"http://{url}/v2/health/ready", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self, url: str = "localhost:8000"):
        """List loaded models"""
        try:
            import requests
            response = requests.get(f"http://{url}/v2/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print("Loaded models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("Failed to get model list.")
        except Exception as e:
            print(f"Error listing models: {e}")

def main():
    parser = argparse.ArgumentParser(description="Triton Inference Server Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "list"],
                       help="Action to perform")
    parser.add_argument("--models-dir", default="models/triton_models",
                       help="Directory containing Triton models")
    parser.add_argument("--http-port", type=int, default=8000,
                       help="HTTP port for Triton server")
    parser.add_argument("--grpc-port", type=int, default=8001,
                       help="GRPC port for Triton server")
    parser.add_argument("--url", default="localhost:8000",
                       help="Triton server URL for status checks")

    args = parser.parse_args()

    manager = TritonManager(args.models_dir)

    if args.action == "start":
        success = manager.start_server(args.http_port, args.grpc_port)
        if success:
            print("\nTriton server started successfully!")
            print(f"HTTP endpoint: http://localhost:{args.http_port}")
            print(f"GRPC endpoint: localhost:{args.grpc_port}")
            print("\nTo test the server:")
            print(f"curl http://localhost:{args.http_port}/v2/health/ready")
            print("\nTo stop the server:")
            print("python triton_manager.py stop")
        else:
            print("Failed to start Triton server.")
            sys.exit(1)

    elif args.action == "stop":
        manager.stop_server()

    elif args.action == "status":
        if manager.check_server_health(args.url):
            print("Triton server is healthy and ready.")
        else:
            print("Triton server is not responding.")

    elif args.action == "list":
        manager.list_models(args.url)

if __name__ == "__main__":
    main()