import pytest
from src.bo.models import BOModel
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
)
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun


@pytest.fixture
def experiment():
    return get_branin_experiment(with_trial=True)


@pytest.fixture
def data(experiment):
    return get_branin_data(trials=[experiment.trials[0]])


class Test_BOModel:
    def test_bo_model(self, experiment, data):
        bo_model = BOModel(experiment, data)
        bo_model.init_model()
        for i in range(2):
            generator_run = bo_model.gen(n=3)
            print(generator_run.arms)
            # format like [Arm(parameters={'x1': -5.0, 'x2': 0.0}), Arm(parameters={'x1': -5.0, 'x2': 15.0}), Arm(parameters={'x1': 10.0, 'x2': 0.0})]
            candidates = [arm.parameters for arm in generator_run.arms]
            print(candidates)
            # format like [{'x1': -4, 'x2': 0.0}, {'x1': 10.0, 'x2': 4}, {'x1': -5.0, 'x2': 15.0}]
            # TODO
            # add llm filtering logic
            # llm = ReasoningModel()
            # filtered_arms = llm.filter(candidates)
            # print(filtered_arms)
            # format like [{'x1': -4, 'x2': 0.0}]
            filtered_candidates = [
                Arm(parameters=params) for params in candidates
            ]
            filtered_generator_run = GeneratorRun(arms=filtered_candidates)
            trial = experiment.new_batch_trial(
                generator_run=filtered_generator_run
            )
            trial.run()
            trial.mark_completed()
            assert candidates is not None
            assert generator_run.arms is not None
            assert filtered_candidates is not None
        assert bo_model.model_bridge is not None
