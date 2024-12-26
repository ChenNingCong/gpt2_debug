# %load_ext autoreload
# %autoreload 2
import tiny_trainer
from test_trainer import *
from model import *
from omegaconf import DictConfig, OmegaConf
import hydra

# os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"]="1" 
# os.environ["TORCHINDUCTOR_CACHE_DIR"]="/home/nchen3/nchen3/llm_test/llm.c/nchen3/ir_cache/"
from hydra import *
if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="config-debug-no-shuffle")
        global trainer
        trainer = TestTrainer(factory=TestFactory(), config=cfg.trainer)
        print(OmegaConf.to_yaml(cfg))
        trainer.run()