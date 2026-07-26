"""Tests for load_policy_from_checkpoint utility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from lerobot_edge.core.utils import load_policy_from_checkpoint


class SimplePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(7, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(next(iter(batch.values())))

    def reset(self) -> None:
        pass


class TestLoadPolicyFromCheckpoint:
    """Test load_policy_from_checkpoint with mocked dependencies."""

    @patch("lerobot_edge.core.utils.importlib.import_module")
    def test_from_pretrained_success(self, mock_import):
        mock_cls = MagicMock()
        mock_model = SimplePolicy()
        mock_cls.from_pretrained.return_value = mock_model
        mock_mod = MagicMock()
        mock_mod.SmolVLAPolicy = mock_cls
        mock_import.return_value = mock_mod

        result = load_policy_from_checkpoint("lerobot/smolvla_base", "smolvla", "cpu")
        assert result is mock_model
        mock_cls.from_pretrained.assert_called_once_with("lerobot/smolvla_base")
        assert result.training is False

    def test_from_pretrained_failure_falls_back_to_factory(self):
        mock_config = MagicMock()
        mock_config.pretrained_path = None
        mock_config.device = None
        mock_model = SimplePolicy()

        mock_import_mod = MagicMock(side_effect=ImportError("SmolVLA not installed"))
        mock_factory_config = MagicMock(return_value=mock_config)
        mock_factory_make = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "lerobot.policies.factory": MagicMock(
                    make_policy_config=mock_factory_config, make_policy=mock_factory_make
                )
            },
        ), patch(
            "lerobot_edge.core.utils.importlib.import_module", side_effect=mock_import_mod
        ):
            result = load_policy_from_checkpoint("some/path", "smolvla", "cpu")
            assert result is mock_model

    def test_from_pretrained_raises_falls_back_to_factory(self):
        mock_config = MagicMock()
        mock_config.pretrained_path = None
        mock_config.device = None
        mock_model = SimplePolicy()

        mock_factory_config = MagicMock(return_value=mock_config)
        mock_factory_make = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "lerobot.policies.factory": MagicMock(
                    make_policy_config=mock_factory_config, make_policy=mock_factory_make
                )
            },
        ), patch(
            "lerobot_edge.core.utils.importlib.import_module",
            side_effect=RuntimeError("Connection error"),
        ):
            result = load_policy_from_checkpoint("some/path", "smolvla", "cpu")
            assert result is mock_model

    def test_smolvla_type_tries_from_pretrained_first(self):
        with patch("lerobot_edge.core.utils.importlib.import_module") as mock_import:
            mock_cls = MagicMock()
            mock_model = SimplePolicy()
            mock_cls.from_pretrained.return_value = mock_model
            mock_mod = MagicMock()
            mock_mod.SmolVLAPolicy = mock_cls
            mock_import.return_value = mock_mod

            load_policy_from_checkpoint("test/checkpoint", "smolvla", "cpu")
            mock_import.assert_called_with("lerobot.policies.smolvla.modeling_smolvla")

    def test_non_smolvla_type_goes_to_factory(self):
        mock_config = MagicMock()
        mock_config.pretrained_path = None
        mock_config.device = None
        mock_model = SimplePolicy()

        with patch(
            "lerobot.policies.factory.make_policy_config", return_value=mock_config
        ) as mock_make_config, patch(
            "lerobot.policies.factory.make_policy", return_value=mock_model
        ):
            result = load_policy_from_checkpoint("test/path", "act", "cpu")
            mock_make_config.assert_called_once_with("act")
            assert mock_config.pretrained_path == "test/path"
            assert mock_config.device == "cpu"
            assert result is mock_model

    def test_model_is_in_eval_mode(self):
        mock_cls = MagicMock()
        mock_model = SimplePolicy()
        mock_model.train()
        mock_cls.from_pretrained.return_value = mock_model
        mock_mod = MagicMock()
        mock_mod.SmolVLAPolicy = mock_cls

        with patch("lerobot_edge.core.utils.importlib.import_module", return_value=mock_mod):
            result = load_policy_from_checkpoint("test", "smolvla", "cpu")
            assert result.training is False

    def test_from_pretrained_returns_nn_module(self):
        mock_cls = MagicMock()
        mock_model = SimplePolicy()
        mock_cls.from_pretrained.return_value = mock_model
        mock_mod = MagicMock()
        mock_mod.SmolVLAPolicy = mock_cls

        with patch("lerobot_edge.core.utils.importlib.import_module", return_value=mock_mod):
            result = load_policy_from_checkpoint("test", "smolvla", "cpu")
            assert isinstance(result, nn.Module)

    def test_factory_fallback_calls_to_device(self):
        mock_config = MagicMock()
        mock_config.pretrained_path = None
        mock_config.device = None
        mock_model = MagicMock(spec=nn.Module)

        mock_factory_config = MagicMock(return_value=mock_config)
        mock_factory_make = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "lerobot.policies.factory": MagicMock(
                    make_policy_config=mock_factory_config, make_policy=mock_factory_make
                )
            },
        ):
            load_policy_from_checkpoint("test", "act", "cpu")
            mock_model.to.assert_called_once_with("cpu")
            mock_model.eval.assert_called_once()

    def test_known_arch_not_in_dict_goes_to_factory(self):
        mock_config = MagicMock()
        mock_config.pretrained_path = None
        mock_config.device = None
        mock_model = SimplePolicy()

        mock_factory_config = MagicMock(return_value=mock_config)
        mock_factory_make = MagicMock(return_value=mock_model)

        with patch.dict(
            "sys.modules",
            {
                "lerobot.policies.factory": MagicMock(
                    make_policy_config=mock_factory_config, make_policy=mock_factory_make
                )
            },
        ):
            result = load_policy_from_checkpoint("test", "unknown_type", "cpu")
            assert result is mock_model
