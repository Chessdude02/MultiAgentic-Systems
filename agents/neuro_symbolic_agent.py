import numpy as np

from agents.symbolic_override_agent import SymbolicOverrideAgent
from utils.pipeline import latest_window


class NeuroSymbolicAgent:
    def __init__(self, predictor=None, model_dir="models"):
        self.predictor = predictor
        self.model_dir = model_dir
        self.symbolic = SymbolicOverrideAgent()

    def fuse_predictions(self, raw, symbolic):
        return [(r * 0.7 + s * 0.3) for r, s in zip(raw, symbolic)]

    def explain(self, raw, symbolic, fused):
        explanations = []
        for r, s, f in zip(raw, symbolic, fused):
            if abs(r - s) > 1e-9:
                explanations.append(f"Adjusted from {r:.2f} to {f:.2f} using RSI/MACD rule.")
            else:
                explanations.append(f"Used model prediction: {f:.2f}")
        return explanations

    def _get_predictor(self):
        if self.predictor is None:
            from agents.prediction_agent import PredictionAgent
            self.predictor = PredictionAgent().load_models(self.model_dir)
        if not self.predictor.has_models():
            raise FileNotFoundError(
                f"No trained models in '{self.model_dir}/'. Run `python train.py` first."
            )
        return self.predictor

    def predict_detailed(self, df):
        """Next-period close from the model ensemble, adjusted by symbolic rules.

        ``df`` must already contain the FeatureEngineerAgent indicators.
        """
        predictor = self._get_predictor()
        last_close = float(df["Close"].iloc[-1])

        returns = {k: float(v[0]) for k, v in predictor.predict(latest_window(df)).items()}
        raw = last_close * (1 + float(np.mean(list(returns.values()))))

        symbolic = self.symbolic.override(df.iloc[[-1]], [raw])[0]
        fused = self.fuse_predictions([raw], [symbolic])[0]
        return {
            "last_close": last_close,
            "model_returns": returns,
            "raw": raw,
            "symbolic": symbolic,
            "prediction": round(fused, 2),
            "explanation": self.explain([raw], [symbolic], [fused])[0],
        }

    def predict(self, df):
        return self.predict_detailed(df)["prediction"]
