# Ray RLlib Deep Dive

Ray RLlib is the distributed RL library that powers TensorTrade training. This tutorial explains how to configure and use it effectively.

## Learning Objectives

After this tutorial, you will understand:
- How Ray RLlib works
- Key configuration options for trading
- Custom callbacks for tracking
- Distributed training setup

---

## What is Ray RLlib?

Ray RLlib is a scalable RL library that:
- Implements many RL algorithms (PPO, DQN, A2C, etc.)
- Supports distributed training across CPUs/GPUs
- Provides callbacks for custom metrics
- Integrates with Gym/Gymnasium environments

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ray RLlib                               │
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐   │
│   │                    Algorithm (PPO)                     │   │
│   │  - Policy network                                      │   │
│   │  - Training loop                                       │   │
│   │  - Experience collection                               │   │
│   └───────────────────────────────────────────────────────┘   │
│                            │                                    │
│           ┌────────────────┼────────────────┐                  │
│           │                │                │                  │
│           v                v                v                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐            │
│   │ Worker 1  │    │ Worker 2  │    │ Worker N  │            │
│   │ (env copy)│    │ (env copy)│    │ (env copy)│            │
│   └───────────┘    └───────────┘    └───────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setting Up Ray

### Initialization

```python
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray (local mode)
ray.init(
    num_cpus=6,              # Use 6 CPU cores
    ignore_reinit_error=True, # Don't error if already initialized
    log_to_driver=False      # Reduce log verbosity
)

# Register custom environment
register_env("TradingEnv", create_env)
```

### Environment Factory

RLlib needs a factory function to create environments:

```python
def create_env(config: Dict[str, Any]):
    """Factory function for TradingEnv."""
    data = pd.read_csv(config["csv_filename"])

    price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")

    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=config.get("commission", 0.001))
    )(price)

    cash = Wallet(exchange, config.get("initial_cash", 10000) * USD)
    asset = Wallet(exchange, 0 * BTC)
    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(data[c]), dtype="float").rename(c)
                for c in config.get("feature_cols", [])]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.5)
    )
    env.portfolio = portfolio  # Store for callbacks
    return env
```

---

## PPO Configuration

### Complete Example

```python
config = (
    PPOConfig()
    # Environment
    .environment(
        env="TradingEnv",
        env_config={
            "csv_filename": "/path/to/data.csv",
            "feature_cols": ["ret_1h", "rsi", "trend"],
            "window_size": 17,
            "max_allowed_loss": 0.32,
            "commission": 0.003,
            "initial_cash": 10000,
        }
    )

    # Framework
    .framework("torch")  # Use PyTorch

    # Rollout workers
    .env_runners(
        num_env_runners=4,  # Parallel environments
        rollout_fragment_length=200,
    )

    # Callbacks
    .callbacks(MyCallbacks)

    # Training hyperparameters
    .training(
        # Learning
        lr=3.29e-05,
        gamma=0.992,
        lambda_=0.9,

        # PPO-specific
        clip_param=0.123,
        entropy_coeff=0.015,
        vf_clip_param=100.0,

        # Batching
        train_batch_size=2000,
        sgd_minibatch_size=256,
        num_sgd_iter=7,

        # Network architecture
        model={
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "tanh",
        },
    )

    # Resources
    .resources(
        num_gpus=0,  # Set to 1 for GPU training
    )
)
```

### Key Parameters Explained

#### Learning Parameters

| Parameter | Meaning | Our Value | Why |
|-----------|---------|-----------|-----|
| `lr` | Learning rate | 3.29e-05 | Very low = stable learning |
| `gamma` | Discount factor | 0.992 | High = values future rewards |
| `lambda_` | GAE parameter | 0.9 | Bias-variance tradeoff |

#### PPO Parameters

| Parameter | Meaning | Our Value | Why |
|-----------|---------|-----------|-----|
| `clip_param` | Policy change limit | 0.123 | Moderate clipping |
| `entropy_coeff` | Exploration bonus | 0.015 | Low = mostly exploit |
| `vf_clip_param` | Value function clipping | 100.0 | Don't clip value function |

#### Batching Parameters

| Parameter | Meaning | Our Value | Why |
|-----------|---------|-----------|-----|
| `train_batch_size` | Samples per update | 2000 | Moderate batch |
| `sgd_minibatch_size` | Mini-batch size | 256 | Standard size |
| `num_sgd_iter` | SGD passes per batch | 7 | Multiple passes |

---

## Custom Callbacks

Callbacks let you track custom metrics during training.

### Basic Callback

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class TradingCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker, base_env, policies,
                         episode, env_index=None, **kwargs):
        """Called at the start of each episode."""
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            episode.user_data["initial_worth"] = float(env.portfolio.net_worth)

    def on_episode_end(self, *, worker, base_env, policies,
                       episode, env_index=None, **kwargs):
        """Called at the end of each episode."""
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, 'portfolio'):
            final_worth = float(env.portfolio.net_worth)
            initial_worth = episode.user_data.get("initial_worth", 10000)

            # Custom metrics
            pnl = final_worth - initial_worth
            pnl_pct = (pnl / initial_worth) * 100

            episode.custom_metrics["pnl"] = pnl
            episode.custom_metrics["pnl_pct"] = pnl_pct
            episode.custom_metrics["final_worth"] = final_worth
```

### Using Custom Callbacks

```python
config = (
    PPOConfig()
    ...
    .callbacks(TradingCallbacks)
)
```

### Accessing Metrics

```python
result = algo.train()

# Metrics from callbacks
custom_metrics = result.get('env_runners', {}).get('custom_metrics', {})
avg_pnl = custom_metrics.get('pnl_mean', 0)
avg_pnl_pct = custom_metrics.get('pnl_pct_mean', 0)

print(f"Average P&L: ${avg_pnl:+,.0f} ({avg_pnl_pct:+.1f}%)")
```

---

## Training Loop

### Basic Loop

```python
algo = config.build()

for i in range(100):
    result = algo.train()

    # Get metrics
    reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
    custom = result.get('env_runners', {}).get('custom_metrics', {})
    pnl = custom.get('pnl_mean', 0)

    print(f"Iter {i+1}: Reward {reward:.1f}, P&L ${pnl:+,.0f}")
```

### With Validation

```python
algo = config.build()
best_val_pnl = float('-inf')

for i in range(100):
    result = algo.train()

    # Evaluate on validation every 10 iterations
    if (i + 1) % 10 == 0:
        val_pnl = evaluate(algo, val_data)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            algo.save('/tmp/best_model')
            print(f"Iter {i+1}: Val ${val_pnl:+,.0f} *NEW BEST*")
        else:
            print(f"Iter {i+1}: Val ${val_pnl:+,.0f}")

# Load best model for testing
algo.restore('/tmp/best_model')
```

---

## Evaluation

### Manual Evaluation

```python
def evaluate(algo, data: pd.DataFrame, n_episodes: int = 10) -> float:
    """Run n episodes and return average P&L."""
    csv_path = '/tmp/eval.csv'
    data.to_csv(csv_path, index=False)

    env_config = {
        "csv_filename": csv_path,
        "feature_cols": feature_cols,
        # ... other config
    }

    pnls = []
    for _ in range(n_episodes):
        env = create_env(env_config)
        obs, _ = env.reset()
        done = truncated = False

        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)

        pnl = env.portfolio.net_worth - 10000
        pnls.append(pnl)

    return np.mean(pnls)
```

### Using RLlib's Built-in Evaluation

```python
config = (
    PPOConfig()
    ...
    .evaluation(
        evaluation_interval=10,  # Evaluate every 10 iterations
        evaluation_num_episodes=5,
        evaluation_config={
            "env_config": val_env_config,
        }
    )
)

# Now result includes evaluation metrics
result = algo.train()
eval_reward = result.get('evaluation', {}).get('episode_reward_mean', 0)
```

---

## Distributed Training

### Multi-CPU

```python
ray.init(num_cpus=16)

config = (
    PPOConfig()
    ...
    .env_runners(num_env_runners=8)  # 8 parallel workers
)

# Training runs on 8 environments simultaneously
```

### GPU Training

```python
config = (
    PPOConfig()
    ...
    .resources(num_gpus=1)  # Use 1 GPU for policy network
)
```

### Cluster Training

```python
# Connect to Ray cluster
ray.init(address="auto")

# Workers will be distributed across cluster nodes
config = (
    PPOConfig()
    ...
    .env_runners(num_env_runners=32)  # Many workers across cluster
)
```

---

## Saving and Loading

### Save Model

```python
# Save checkpoint
checkpoint_path = algo.save('/tmp/checkpoints')
print(f"Saved to: {checkpoint_path}")

# Save with custom name
checkpoint_path = algo.save('/tmp/my_model')
```

### Load Model

```python
# Load checkpoint
algo.restore('/tmp/checkpoints/checkpoint_000050')

# Or restore from path
from ray.rllib.algorithms.ppo import PPO
algo = PPO(config=config)
algo.restore('/tmp/my_model')
```

### Export Policy

```python
# Export to ONNX for deployment
policy = algo.get_policy()
policy.export_model('/tmp/model_onnx')
```

---

## Common Issues

### Memory Issues

```python
# Reduce batch size
.training(
    train_batch_size=1000,  # Smaller
    sgd_minibatch_size=128,
)

# Reduce workers
.env_runners(num_env_runners=2)
```

### Slow Training

```python
# Add more workers
.env_runners(num_env_runners=8)

# Use GPU
.resources(num_gpus=1)
```

### NaN in Training

```python
# Lower learning rate
.training(lr=1e-5)

# Gradient clipping (already enabled by default in PPO)
```

### Environment Not Resetting Properly

```python
# Make sure reset() recreates all state
def create_env(config):
    # Create everything fresh
    price = Stream.source(...)  # New stream
    exchange = Exchange(...)    # New exchange
    # ...
```

---

## Alternative Algorithms

### DQN (for discrete actions)

```python
from ray.rllib.algorithms.dqn import DQNConfig

config = (
    DQNConfig()
    .environment(env="TradingEnv", env_config=env_config)
    .training(
        lr=1e-4,
        gamma=0.99,
        replay_buffer_config={"capacity": 100000},
    )
)
```

### A2C (simpler than PPO)

```python
from ray.rllib.algorithms.a3c import A2CConfig

config = (
    A2CConfig()
    .environment(env="TradingEnv", env_config=env_config)
    .training(
        lr=5e-4,
        gamma=0.99,
    )
)
```

### SAC (for continuous actions)

```python
from ray.rllib.algorithms.sac import SACConfig

# Requires continuous action space
config = (
    SACConfig()
    .environment(env="TradingEnv", env_config=env_config)
    .training(
        lr=3e-4,
        gamma=0.99,
    )
)
```

---

## Key Takeaways

1. **Ray RLlib handles distributed training** - Just set num_env_runners
2. **Environment factory is crucial** - Must create fresh env each call
3. **Custom callbacks track metrics** - P&L, trades, etc.
4. **PPO is the default** - Works well for trading
5. **Always validate** - Use separate validation data

---

## Checkpoint

Before continuing, verify you understand:

- [ ] How to configure PPOConfig
- [ ] What custom callbacks can track
- [ ] How to save and load models
- [ ] How to evaluate on validation data

---

## Next Steps

[03-optuna.md](03-optuna.md) - Automatic hyperparameter optimization
