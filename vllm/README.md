# vLLM OCR Serving

Serve DeepSeek-OCR-2, dots.mocr, and olmOCR on EC2 via vLLM.

## Setup

- AWS CLI configured
- EC2 key pair (default: `ocr-bench`, edit `aws/config.sh`)
- Key at `~/.ssh/ocr-bench.pem`

## Usage

```bash
# launch a model
./vllm/aws/launch.sh deepseek

# send an image
python vllm/client.py data/pages/LiFePO4_zhao2017/10/page_10.png --model deepseek

# tear down
./vllm/aws/terminate.sh deepseek
```

## AWS scripts (`vllm/aws/`)

| Script | Does |
|--------|------|
| `config.sh` | Region, instance type, AMI, key pair |
| `launch.sh <model>` | Spin up g6.xlarge, install vLLM, start serving |
| `terminate.sh <model>` | Kill the instance |
| `terminate.sh all` | Kill all tracked instances |
| `status.sh` | Show instance states |

First `launch.sh` creates a security group (`ocr-bench-vllm`) opening ports 22 and 8001-8003. Instance IPs save to `aws/instances/` for auto-detection.

## Models

| Model | Size | Port | Launch |
|-------|------|------|--------|
| DeepSeek-OCR-2 | 3B | 8001 | `./vllm/aws/launch.sh deepseek` |
| dots.mocr | 3B | 8002 | `./vllm/aws/launch.sh dots` |
| olmOCR-2-7B (FP8) | 8B | 8003 | `./vllm/aws/launch.sh olmocr` |

Weights download on first launch (~5-10 min).

## Client

```bash
# auto-resolves host from tracked instances
python vllm/client.py image.png --model deepseek

# all models
python vllm/client.py image.png --all

# custom prompt
python vllm/client.py image.png --model dots --prompt "Extract tables as markdown"

# manual host
python vllm/client.py image.png --model deepseek --host 3.85.12.100
```

Results save to `vllm/results/`.

## Instance sizing

- **g6.xlarge** (1x L4 24GB) fits any single model — default
- One model per instance for benchmarking isolation

## Logs

```bash
ssh -i ~/.ssh/ocr-bench.pem ubuntu@<IP> 'tail -f ~/vllm_serve.log'
```
