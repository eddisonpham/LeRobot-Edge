# LeRobot API Map (as of lerobot==0.6.0)

## Policy Interface (class, methods, signatures)

### PreTrainedPolicy (base class)
- Inherits from: `nn.Module`, `HubMixin`, `abc.ABC`
- Location: `lerobot.policies.pretrained.PreTrainedPolicy`

**Required methods:**
- `reset() -> None` – Called at the start of each episode to clear state
- `select_action(batch: dict[str, Tensor]) -> Tensor` – Returns next action for inference
- `forward(batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]` – Training forward pass, returns (loss, info)
- `predict_action_chunk(batch: dict[str, Tensor]) -> Tensor` – Returns action sequence
- `get_optim_params() -> dict` – Returns optimizer parameters

**Class attributes (enforced by __init_subclass__):**
- `config_class: type[PreTrainedConfig]` – Must be set
- `name: str` – Must be set

**Inherited methods:**
- `from_pretrained(pretrained_name_or_path, ...)` – Load from Hub or local
- `save_pretrained(save_directory, ...)` – Save to disk

## Plugin/Registration Mechanism

### How it works
LeRobot uses **draccus.ChoiceRegistry** on `PreTrainedConfig`. The registration is done via:

```python
from lerobot.configs import PreTrainedConfig

@PreTrainedConfig.register_subclass("my_policy_name")
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    type: str = "my_policy_name"
    # ... config fields
```

### Discovery flow
1. `get_policy_class(name)` in `lerobot.policies.factory` checks a hardcoded if/elif chain
2. If not found, falls back to `_get_policy_cls_from_policy_name(name)` which:
   - Calls `PreTrainedConfig.get_known_choices()` to find registered configs
   - Uses naming convention: config class `FooConfig` → policy class `FooPolicy`
   - Dynamically imports from `configuration_foo` → `modeling_foo`
3. `make_policy_config(policy_type)` follows the same pattern

### Processor factory
- Each policy type has a `make_{type}_pre_post_processors` function
- Factory falls back to `_make_processors_from_policy_config` for third-party plugins
- Naming convention: `configuration_foo` → `processor_foo` → `make_foo_pre_post_processors`

### Working toy example
```python
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
import torch.nn as nn

@PreTrainedConfig.register_subclass("edge_test")
@dataclass
class EdgeTestConfig(PreTrainedConfig):
    type: str = "edge_test"
    # Required abstract properties
    @property
    def observation_delta_indices(self): return None
    @property
    def action_delta_indices(self): return None
    @property
    def reward_delta_indices(self): return None
    def get_optimizer_preset(self): return {"optimizer_cls": "AdamW", "lr": 1e-4}
    def get_scheduler_preset(self): return None
    def validate_features(self): pass

class EdgeTestPolicy(PreTrainedPolicy):
    config_class = EdgeTestConfig
    name = "edge_test"
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.linear = nn.Linear(7, 2)
    
    def select_action(self, batch):
        return self.linear(batch["observation.state"])
    
    def forward(self, batch):
        return torch.tensor(0.0), {}
    
    def predict_action_chunk(self, batch):
        return self.select_action(batch)
    
    def reset(self): pass
    def get_optim_params(self): return {"params": list(self.parameters())}
```

## Eval/Record Checkpoint Loading Path

- `lerobot-eval` uses `make_policy(cfg)` which:
  1. Calls `get_policy_class(cfg.type)` to find the policy class
  2. Infers input/output features from dataset metadata or env config
  3. Calls `policy_cls.from_pretrained(...)` if `cfg.pretrained_path` is set
  4. Or `policy_cls(config=cfg)` for fresh initialization
- Third-party `--policy.type` values work via the fallback in `get_policy_class()`

## Config System and Third-Party Fields

- **Draccus** (v0.10.0): dataclass-based config system
- `PreTrainedConfig` inherits from `draccus.ChoiceRegistry`
- Any field defined in your config dataclass becomes a CLI flag
- Example: `--policy.deploy_backend=onnx_int8` works automatically

## Sim Benchmarks Available

- **PushT** – Lightest weight, 2D physics, no GPU required (primary dev-loop)
- **MetaWorld** – Medium complexity, 3D manipulation tasks
- **LIBERO** – Heavier, MuJoCo-based, good for periodic sanity checks
- **RoboTwin 2.0**, **RoboCasa365** – Available in v0.6.0

## Open Questions / Risks

1. **Processor factory naming**: Must follow convention exactly (`configuration_X` → `processor_X`)
2. **Action chunking**: Some policies return action sequences, not single actions
3. **Feature validation**: `validate_features()` must accept the features LeRobot passes
4. **PEFT integration**: Built-in support, but not needed for deployment variants
