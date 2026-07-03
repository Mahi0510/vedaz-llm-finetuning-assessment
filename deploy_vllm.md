# Reference Deployment Guide: vLLM + Docker + Nginx
===================================================

**Author: Mahika Morolia**  
*B.Tech Computer Science Engineering*  
*Specialization: Cybersecurity & Forensics*

This document provides instructions to deploy the fine-tuned **Qwen2.5-7B Vedic Astrologer** model in a secure, high-throughput, and deployment-ready architecture environment.

---

## 1. Architecture Overview

The production architecture is designed to handle thousands of concurrent queries under 100ms latency:

```
[Web/Client Chat] ---> [Nginx HTTPS (Port 443)] ---> [vLLM OpenAI Server (Port 8000)]
                              |
                              v
                 [Systemd / Docker Monitor]
```

- **Inference Engine**: vLLM (OpenAI API compliant)
- **Containerization**: Docker & NVIDIA Container Toolkit (CUDA 12.1+)
- **Security & Reverse Proxy**: Nginx with SSL (Let's Encrypt)
- **Orchestration**: Systemd (reboots on crash)
- **Monitoring**: Prometheus and Grafana (built-in vLLM metrics)

---

## 2. Prerequisites & Server Setup

Ensure your host meets the minimum system requirements:
- **GPU**: NVIDIA A10G, L4, A100, or RTX 3090/4090 (Minimum 16GB VRAM, 24GB+ recommended)
- **Drivers**: CUDA 12.1+ compatible drivers
- **Disk**: 50GB NVMe SSD space

### Install NVIDIA Container Toolkit
For Docker to communicate with the host GPU, register the runtime driver wrapper:

```bash
# Add package repositories
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Restart docker daemon
sudo systemctl restart docker
```

---

## 3. Container Deployment via Docker Compose

We containerize our serving layer using vLLM's official registry images. 

### Step 1: Create `docker-compose.yml`
Save the following configuration inside the repository:

```yaml
version: '3.8'

services:
  vllm-astrologer:
    image: vllm/vllm-openai:v0.6.1
    container_name: vllm-astrologer-engine
    environment:
      - HF_TOKEN=${HF_TOKEN}  # Hugging Face authorization key if private
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./outputs/merged_qwen2.5-7b-vedic-astrologer:/model/fused_weights
    ports:
      - "8000:8000"
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model /model/fused_weights
      --port 8000
      --max-model-len 2048
      --gpu-memory-utilization 0.90
      --trust-remote-code
    restart: always
```

### Step 2: Spin Up the Container
Execute the following to run the inference service in the background:

```bash
docker-compose up -d
```

---

## 4. HTTPS and Nginx Gateway Configuration

To secure communication from our Gradio/React frontend clients, route incoming API traffic through Nginx with SSL configuration.

### Create Nginx Server Block
Create `/etc/nginx/sites-available/astrologer.conf`:

```nginx
server {
    listen 80;
    server_name api.vedazastrology.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name api.vedazastrology.org;

    ssl_certificate /etc/letsencrypt/live/api.vedazastrology.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.vedazastrology.org/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

Enable the configuration and reload:

```bash
sudo ln -s /etc/nginx/sites-available/astrologer.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 5. System Daemon Management (Systemd)

To run the engine directly as a background service on raw instances (alternative to Docker), set up a Systemd service file `/etc/systemd/system/vllm.service`:

```ini
[Unit]
Description=vLLM OpenAI Server for Vedic Astrologer
After=network.target

[Service]
Type=simple
User=ubuntu
Environment="CUDA_VISIBLE_DEVICES=0"
WorkingDirectory=/home/ubuntu/astrology-repo
ExecStart=/home/ubuntu/miniconda3/envs/vllm/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model outputs/merged_qwen2.5-7b-vedic-astrologer \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
Restart=on-failure
RestartSec=10
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

To load and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm.service
sudo systemctl start vllm.service
```

Check status:

```bash
sudo systemctl status vllm.service
```

---

## 6. Real-time Monitoring and Infrastructure Checks

vLLM exposes standard metrics out-of-the-box in Prometheus format on `http://localhost:8000/metrics`.

### Key Metrics to Monitor
- **`vllm:num_requests_waiting`**: Number of requests queued. If >5 regularly, scale GPUs.
- **`vllm:gpu_cache_usage_factor`**: Memory load on KV cache. Keep below 0.95.
- **`vllm:num_requests_running`**: Concurrent sequences being computed in parallel.
- **`vllm:request_generation_tokens_per_second`**: Model throughput rate. Target: >40 tokens/sec.

*Ready for deployment-ready reference architectures.*
