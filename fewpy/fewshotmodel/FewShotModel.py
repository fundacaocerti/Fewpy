class FewShotModel:

    def __init__(self, model: str) -> None:
        self.model = self.load_model()

    def load_model(model: str) -> FSLModel:
        ...

    def predict(self, in_features, output_logits: bool = False):
        # TODO - verify input

        # generate output
        return self.model(in_features)
        