"""Unit tests for Hydra configurations."""
import pytest
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


class TestConfigs:
    """Test cases for Hydra configurations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup and teardown for each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()
    
    def get_config_dir(self):
        """Get the absolute path to configs directory."""
        return str(Path(__file__).parent.parent.parent / "configs")
    
    def test_eval_config_exists(self):
        """Test that eval.yaml config exists."""
        config_path = Path(self.get_config_dir()) / "eval.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_infer_config_exists(self):
        """Test that infer.yaml config exists."""
        config_path = Path(self.get_config_dir()) / "infer.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_eval_config_loads(self):
        """Test that eval config loads correctly."""
        with initialize_config_dir(config_dir=self.get_config_dir(), version_base="1.3"):
            cfg = compose(config_name="eval")
            
            assert cfg is not None
            assert "checkpoint_path" in cfg
            assert "trainer" in cfg
            assert "output_dir" in cfg
    
    def test_infer_config_loads(self):
        """Test that infer config loads correctly."""
        with initialize_config_dir(config_dir=self.get_config_dir(), version_base="1.3"):
            cfg = compose(config_name="infer")
            
            assert cfg is not None
            assert "checkpoint_path" in cfg
            assert "input_folder" in cfg
            assert "output_folder" in cfg
            assert "max_images" in cfg
    
    def test_paths_config_exists(self):
        """Test that paths/default.yaml config exists."""
        config_path = Path(self.get_config_dir()) / "paths" / "default.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_model_config_exists(self):
        """Test that model/dog_breed.yaml config exists."""
        config_path = Path(self.get_config_dir()) / "model" / "dog_breed.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_data_config_exists(self):
        """Test that data/dog_breed.yaml config exists."""
        config_path = Path(self.get_config_dir()) / "data" / "dog_breed.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
    
    def test_eval_config_has_required_fields(self):
        """Test that eval config has all required fields."""
        with initialize_config_dir(config_dir=self.get_config_dir(), version_base="1.3"):
            cfg = compose(config_name="eval")
            
            # Check trainer settings
            assert "accelerator" in cfg.trainer
            assert "devices" in cfg.trainer
            
            # Check output settings
            assert "metrics_file" in cfg
    
    def test_infer_config_has_required_fields(self):
        """Test that infer config has all required fields."""
        with initialize_config_dir(config_dir=self.get_config_dir(), version_base="1.3"):
            cfg = compose(config_name="infer")
            
            # Check image settings
            assert "image_extensions" in cfg
            assert "max_images" in cfg
            
            # Check visualization settings
            assert "save_visualization" in cfg
            assert "visualization_cols" in cfg
