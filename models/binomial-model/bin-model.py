import math
import numpy as np

class BinomialOption:
    def __init__(self, S0, K, T, r, sigma, N):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.delta_t = T / N
        self.u = math.exp(sigma * math.sqrt(self.delta_t))
        self.v = 1 / self.u
        self.p = (math.exp(r * self.delta_t) - self.v) / (self.u - self.v)
        self.discount = math.exp(-r * self.delta_t)

    def _stock_prices(self):
        j = np.arange(self.N + 1)
        return self.S0 * (self.u ** j) * (self.v ** (self.N - j))

    def _backtrack(self, payoffs):
        V = payoffs.copy()
        for i in range(self.N - 1, -1, -1):
            V = self.discount * (self.p * V[1:] + (1 - self.p) * V[:-1])
        return V[0]

    def price_call(self):
        ST = self._stock_prices()
        payoff = np.maximum(ST - self.K, 0)
        return self._backtrack(payoff)

    def price_put(self):
        ST = self._stock_prices()
        payoff = np.maximum(self.K - ST, 0)
        return self._backtrack(payoff)
