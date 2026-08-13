class NeuroSymbolicAgent:
    def fuse_predictions(self, raw, symbolic):
        return [(r * 0.7 + s * 0.3) for r, s in zip(raw, symbolic)]

    def explain(self, raw, symbolic, fused):
        explanations = []
        for r, s, f in zip(raw, symbolic, fused):
            if abs(r - s) > 2.0:
                explanations.append(f"Adjusted from {r:.2f} to {f:.2f} using rule.")
            else:
                explanations.append(f"Used model prediction: {f:.2f}")
        return explanations

    def predict(self, df):
        # Basic prediction logic (replace with your model output if needed)
        last_close = df["Close"].iloc[-1]
        return round(last_close * 1.01, 2)
