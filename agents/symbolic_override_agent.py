class SymbolicOverrideAgent:
    def override(self, df, predictions):
        adjusted = []
        rsi = df["RSI"]
        macd_line = df["MACD"]
        signal_line = df["Signal"]

        for i in range(len(predictions)):
            pred = float(predictions[i])
            try:
                r = float(rsi.iloc[i])
                macd = float(macd_line.iloc[i])
                signal = float(signal_line.iloc[i])

                if r < 30 and macd > signal:
                    adjusted.append(pred * 1.02)
                elif r > 70 and macd < signal:
                    adjusted.append(pred * 0.98)
                else:
                    adjusted.append(pred)
            except Exception as e:
                print(f"Warning at index {i}: {e}")
                adjusted.append(pred)

        return adjusted
