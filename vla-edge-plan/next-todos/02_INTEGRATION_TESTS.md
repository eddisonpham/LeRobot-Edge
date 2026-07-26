# 02 — Integration Tests

> Priority: P0 (milestone M2 acceptance criterion)
> Estimated effort: 4-8 hours

## 2.1 End-to-end smoke test with real SmolVLA

**Problem:** The user explicitly asked for smoke testing, and milestone M2 acceptance criteria require: *"install both packages fresh, load smolvla_base from the Hub, quantize, export to ONNX, run 2 episodes through lerobot-eval, assert it completes."* The current test suite is all unit tests with synthetic models.

**Subtasks:**
- [ ] Create `tests/test_integration.py` with end-to-end smoke tests
- [ ] Test: Load SmolVLA from Hub, wrap in CompressedPolicy (identity backend)
- [ ] Test: Run `select_action` on real SmolVLA input
- [ ] Test: Quantize SmolVLA with dynamic INT8
- [ ] Test: Export quantized SmolVLA to ONNX
- [ ] Test: Run inference through ONNX Runtime backend
- [ ] Test: Compare output shapes between FP32 and quantized variants
- [ ] Add `@pytest.mark.integration` marker for slow tests
- [ ] Add `@pytest.mark.slow` marker for tests requiring model download

**Acceptance criteria:**
- Integration test loads real SmolVLA checkpoint
- Integration test runs quantization successfully
- Integration test exports to ONNX successfully
- Integration test runs inference through ONNX Runtime
- Output shapes match between variants

## 2.2 PushT evaluation integration

**Problem:** No test verifies that the edge policy variants work with `lerobot-eval` on PushT.

**Subtasks:**
- [ ] Create `tests/test_eval_integration.py`
- [ ] Test: Run `lerobot-eval` with `edge_identity` policy on PushT
- [ ] Test: Run `lerobot-eval` with `edge_quant_int8` policy on PushT
- [ ] Test: Verify success rate is non-zero
- [ ] Test: Verify completion within reasonable time
- [ ] Mark as slow/integration test

**Acceptance criteria:**
- `lerobot-eval --policy.type=edge_identity --env.type=pusht` completes
- Success rate is recorded
- Test completes within 5 minutes

## 2.3 Plugin registration verification

**Problem:** No test verifies that edge policies are discoverable by LeRobot's factory.

**Subtasks:**
- [ ] Test: `PreTrainedConfig.get_known_choices()` contains all edge variants
- [ ] Test: `get_policy_class("edge_identity")` returns correct class
- [ ] Test: `get_policy_class("edge_quant_int8")` returns correct class
- [ ] Test: `make_policy_config("edge_identity")` returns correct config
- [ ] Test: Policy can be instantiated from config

**Acceptance criteria:**
- All 6 edge variants are discoverable via LeRobot's registry
- Policy classes can be instantiated from configs
