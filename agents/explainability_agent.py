import shap

class ExplainabilityAgent:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.Explainer(model)

    def explain(self, X):
        shap_values = self.explainer(X)
        return shap_values
