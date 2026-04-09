# Lunar Landing

## Documentation
- [Lunar Landing.pdf](Lunar%20Landing.pdf)
- [3. Lunar landing.docx](3.%20Lunar%20landing.docx)

## Dataset
- Built-in Gymnasium environment (no download needed)
- **Environment**: Gymnasium LunarLander-v3

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
