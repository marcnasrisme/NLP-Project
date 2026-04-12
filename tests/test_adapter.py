import pytest

from src.models.adapter import load_config, make_lora_config, make_training_args


@pytest.fixture()
def config():
    return load_config("configs/adapter_training.yaml")


class TestLoadConfig:
    def test_has_required_sections(self, config):
        assert "model" in config
        assert "lora" in config
        assert "training" in config

    def test_model_name(self, config):
        assert config["model"]["name"] == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_quantization(self, config):
        assert config["model"]["quantization"] == "4bit"


class TestLoraConfig:
    def test_rank(self, config):
        lora = make_lora_config(config)
        assert lora.r == 16

    def test_alpha(self, config):
        lora = make_lora_config(config)
        assert lora.lora_alpha == 32

    def test_target_modules(self, config):
        lora = make_lora_config(config)
        assert set(lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_task_type(self, config):
        lora = make_lora_config(config)
        assert lora.task_type == "CAUSAL_LM"

    def test_dropout(self, config):
        lora = make_lora_config(config)
        assert lora.lora_dropout == 0.05

    def test_no_bias(self, config):
        lora = make_lora_config(config)
        assert lora.bias == "none"


class TestTrainingArgs:
    def test_epochs(self, config):
        args = make_training_args(config, "test_output")
        assert args.num_train_epochs == 3

    def test_batch_size(self, config):
        args = make_training_args(config, "test_output")
        assert args.per_device_train_batch_size == 4

    def test_learning_rate(self, config):
        args = make_training_args(config, "test_output")
        assert args.learning_rate == 2.0e-4

    def test_gradient_accumulation(self, config):
        args = make_training_args(config, "test_output")
        assert args.gradient_accumulation_steps == 4

    def test_save_steps(self, config):
        args = make_training_args(config, "test_output")
        assert args.save_steps == 100

    def test_output_dir(self, config):
        args = make_training_args(config, "my/output/dir")
        assert args.output_dir == "my/output/dir"

    def test_save_total_limit(self, config):
        args = make_training_args(config, "test_output")
        assert args.save_total_limit == 2

    def test_fp16(self, config):
        args = make_training_args(config, "test_output")
        assert args.fp16 is True

    def test_max_length(self, config):
        args = make_training_args(config, "test_output")
        assert args.max_length == 512
