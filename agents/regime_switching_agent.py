from sklearn.cluster import KMeans

class RegimeSwitchingAgent:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters)

    def detect(self, close_prices):
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(5).std().dropna()
        labels = self.kmeans.fit_predict(volatility.values.reshape(-1, 1))
        return labels, volatility
