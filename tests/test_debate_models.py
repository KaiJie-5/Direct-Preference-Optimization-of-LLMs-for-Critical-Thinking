from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from debate.agents import TransformersChatAgent
from debate.config import AgentConfig, GenerationConfig, debate_config_from_mapping, load_debate_config


def test_llama_qwen_config_uses_separate_gpus_and_four_code_blocks() -> None:
    repo_root = Path(__file__).parents[1]
    config = load_debate_config(
        repo_root / "configs" / "multi_agent_debate_llama_qwen.json"
    )

    assert [agent.id for agent in config.agents] == ["llama_70b", "qwen_72b"]
    assert [agent.device_map for agent in config.agents] == [0, 1]
    assert [agent.torch_dtype for agent in config.agents] == ["bfloat16", "bfloat16"]
    assert config.agents[0].model_path == (
        "/iridisfs/scratch/kjl1a21/DPO/models/teacher/Llama-3.3-70B-Instruct"
    )
    assert config.agents[0].prompt_path.name == "llama_70b_debate_placeholder.txt"
    assert config.generation.do_sample is False
    assert config.generation.temperature == 0.0
    assert config.review_blocks == (
        "wrong_code",
        "descriptive_not_answering_research_question",
        "too_broad_code",
        "useful_analytical_code",
    )
    assert [turn.agent_id for turn in config.turns] == [
        "llama_70b",
        "qwen_72b",
        "llama_70b",
        "qwen_72b",
    ]
    assert config.turns[-1].contributes_to_aggregation is True


def test_device_map_accepts_gpu_indexes_and_rejects_booleans(tmp_path: Path) -> None:
    payload = _minimal_config_payload(tmp_path)
    payload["agents"][0]["device_map"] = {"": 0}
    payload["agents"][1]["device_map"] = "auto"
    config = debate_config_from_mapping(payload, base_dir=tmp_path)
    assert config.agents[0].device_map == {"": 0}
    assert config.agents[1].device_map == "auto"

    payload["agents"][0]["device_map"] = True
    with pytest.raises(ValueError, match="device_map"):
        debate_config_from_mapping(payload, base_dir=tmp_path)


def test_transformers_agent_loads_bfloat16_on_explicit_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs: object) -> object:
            calls["tokenizer"] = (model_path, kwargs)
            return SimpleNamespace()

    class FakeLoadedModel:
        def eval(self) -> None:
            calls["eval"] = True

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs: object) -> FakeLoadedModel:
            calls["model"] = (model_path, kwargs)
            return FakeLoadedModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerLoader
    fake_transformers.AutoModelForCausalLM = FakeModelLoader
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    TransformersChatAgent(
        AgentConfig(
            id="llama_70b",
            name="Llama 70B",
            role="Methodologist",
            backend="transformers",
            model_path="/models/llama",
            prompt_path=tmp_path / "prompt.txt",
            torch_dtype="bfloat16",
            device_map=0,
        )
    )

    assert calls["tokenizer"] == ("/models/llama", {"trust_remote_code": False})
    model_path, model_kwargs = calls["model"]
    assert model_path == "/models/llama"
    assert model_kwargs == {
        "torch_dtype": fake_torch.bfloat16,
        "device_map": 0,
        "trust_remote_code": False,
    }
    assert calls["eval"] is True


@pytest.mark.parametrize(
    ("native_pad_token_id", "expected_pad_token_id"),
    [(151643, 151643), (None, 128009)],
)
def test_generation_uses_native_chat_template_eos_and_padding(
    native_pad_token_id: int | None,
    expected_pad_token_id: int,
) -> None:
    calls: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, values: list[int], *, batched: bool = False) -> None:
            self.values = values
            self.shape = (1, len(values)) if batched else (len(values),)

        def to(self, device: str) -> FakeTensor:
            calls.setdefault("input_devices", []).append(device)
            return self

        def __getitem__(self, item: slice) -> FakeTensor:
            return FakeTensor(self.values[item])

    class FakeBatch:
        def __getitem__(self, index: int) -> FakeTensor:
            assert index == 0
            return FakeTensor([10, 11, 12, 20, 21])

    class FakeTokenizer:
        eos_token_id = 128009
        pad_token_id = None

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            calls["chat_template"] = (messages, kwargs)
            return "native-rendered-prompt"

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, FakeTensor]:
            calls["tokenizer_call"] = (prompt, kwargs)
            return {"input_ids": FakeTensor([10, 11, 12], batched=True)}

        def decode(self, token_ids: FakeTensor, **kwargs: object) -> str:
            calls["decode"] = (token_ids.values, kwargs)
            return "generated-json"

    class FakeModel:
        device = "cuda:0"
        generation_config = SimpleNamespace(
            eos_token_id=[128001, 128008, 128009],
            pad_token_id=native_pad_token_id,
        )

        def generate(self, **kwargs: object) -> FakeBatch:
            calls["generate"] = kwargs
            return FakeBatch()

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    agent = object.__new__(TransformersChatAgent)
    agent.agent_id = "llama_70b"
    agent.name = "Llama 70B"
    agent.model_path = "/models/llama"
    agent.torch_dtype = "bfloat16"
    agent.device_map = 0
    agent.trust_remote_code = False
    agent._torch = fake_torch
    agent.tokenizer = FakeTokenizer()
    agent.model = FakeModel()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    result = agent.generate(messages, GenerationConfig(do_sample=False))

    assert calls["chat_template"] == (
        messages,
        {"tokenize": False, "add_generation_prompt": True},
    )
    assert result.rendered_prompt == "native-rendered-prompt"
    assert result.text == "generated-json"
    generation_kwargs = calls["generate"]
    assert generation_kwargs["eos_token_id"] == [128001, 128008, 128009]
    assert generation_kwargs["pad_token_id"] == expected_pad_token_id
    assert generation_kwargs["do_sample"] is False


def _minimal_config_payload(tmp_path: Path) -> dict:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt", encoding="utf-8")
    return {
        "review_pack_path": str(tmp_path),
        "output_root": str(tmp_path),
        "datasets": [{"dataset": "energy"}],
        "agents": [
            {
                "id": "llama_70b",
                "name": "Llama 70B",
                "role": "Methodologist",
                "backend": "dry-run",
                "prompt_path": str(prompt),
            },
            {
                "id": "qwen_72b",
                "name": "Qwen 72B",
                "role": "Auditor",
                "backend": "dry-run",
                "prompt_path": str(prompt),
            },
        ],
        "turns": [
            {"id": "turn1", "agent_id": "llama_70b", "role": "initial"},
            {"id": "turn2", "agent_id": "qwen_72b", "role": "response"},
            {
                "id": "turn3",
                "agent_id": "llama_70b",
                "role": "revision",
                "contributes_to_aggregation": True,
            },
            {
                "id": "turn4",
                "agent_id": "qwen_72b",
                "role": "final",
                "contributes_to_aggregation": True,
            },
        ],
    }
