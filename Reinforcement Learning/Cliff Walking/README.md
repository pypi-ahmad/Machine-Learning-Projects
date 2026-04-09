# Cliff Walking

## Documentation
- [Cliff Walking.pdf](Cliff%20Walking.pdf)
- [9. Cliff walking.docx](9.%20Cliff%20walking.docx)

## Dataset
- Built-in Gymnasium environment (no download needed)
- **Environment**: Gymnasium CliffWalking-v0

## Run
```bash
python run.py                    # full training
python run.py --smoke-test       # quick validation
python run.py --download-only    # download dataset only
python run.py --epochs 10        # custom epochs
python run.py --device cpu       # force CPU
python run.py --no-amp           # disable mixed precision
```

## Metrics
Results in `outputs/metrics.json`:
- avg_reward, success_rate, num_episodes

## Approach
PyTorch DQN
