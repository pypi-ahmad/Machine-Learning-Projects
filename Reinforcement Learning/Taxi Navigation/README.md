# Reinforcement Learning for Taxis

## Documentation
- [Reinforcement Learning for Taxis.pdf](Reinforcement%20Learning%20for%20Taxis.pdf)
- [1. Reinforcement Learning for Taxis.docx](1.%20Reinforcement%20Learning%20for%20Taxis.docx)

## Dataset
- Built-in Gymnasium environment (no download needed)
- **Environment**: Gymnasium Taxi-v3

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
PyTorch DQN + replay buffer
