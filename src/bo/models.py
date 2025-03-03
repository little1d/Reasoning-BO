from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP


class BOModel:
    def __init__(self, experiment, data):
        self.experiment = experiment
        self.data = data
        self.model_bridge = None

    def init_model(self):
        self.model_bridge = Models.BOTORCH_MODULAR(
            experiment=self.experiment,
            data=self.data,
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(botorch_model_class=SingleTaskGP),
                ]
            ),
            botorch_acqf_class=qLogNoisyExpectedImprovement,
        )

    def gen(self, n):
        if self.model_bridge is None:
            raise ValueError(
                "Model has not been initialized. Call initialize_model() first."
            )
        return self.model_bridge.gen(n=n)
