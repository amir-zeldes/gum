from allennlp.common.testing import ModelTestCase


class TestBaseline(ModelTestCase):
    def test_model_can_train(self):
        # This built-in test makes sure that your data can load, that it gets passed to the model
        # correctly, that your model computes a loss in a way that we can get gradients from it,
        # that all of your parameters get non-zero gradient updates, and that we can save and load
        # your model and have the model's predictions remain consistent.
        param_file = "tests/fixtures/config.json"
        self.ensure_model_can_train_save_and_load(param_file, cuda_device=-1)
