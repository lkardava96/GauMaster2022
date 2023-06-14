# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as st
 
from tqdm import tqdm

from pypfopt.expected_returns import mean_historical_return, returns_from_prices
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier, EfficientSemivariance, EfficientCVaR, EfficientCDaR
from pypfopt import BlackLittermanModel
from pypfopt import HRPOpt, CLA
from pypfopt import plotting

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import yfinance as yf
import pypfopt

from sklearn.neural_network import MLPClassifier
# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# define tickers
tickers = ["TSLA", "AMZN", "PEP", "WMT", "SHEL", "DIS", "KER.PA", "AAPL", "NESN.SW", "NFLX"]

ohlc = yf.download(tickers, start="2000-01-01", end="2023-03-31")
prices = ohlc["Adj Close"]
# prices.tail()

market_prices = yf.download("SPY", period="max")["Adj Close"]
# market_prices.head()

mcaps = {}
for t in tickers:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
# mcaps

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)
# delta
# plotting.plot_covariance(S, plot_correlation=True)

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
# market_prior
# market_prior.plot.barh(figsize=(10,5))

# viewdict = {
#     "TSLA": 0.10,
#     "AMZN": 0.30,
#     "PEP": 0.05,
#     "WMT": 0.05,
#     "SHEL": 0.20,
#     "DIS": -0.05,
#     "KER.PA": 0.15,
#     "AAPL": 0.10,
#     "NESN.SW": 0.50,
#     "NFLX": -0.10
# }

# bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

viewdict = {}
for i in prices.columns:
        viewdict[i] = 0
# print(viewdict)

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

# fig, ax = plt.subplots(figsize=(7,7))
# im = ax.imshow(bl.omega)

# # We want to show all ticks...
# ax.set_xticks(np.arange(len(bl.tickers)))
# ax.set_yticks(np.arange(len(bl.tickers)))

# ax.set_xticklabels(bl.tickers)
# ax.set_yticklabels(bl.tickers)
# plt.show()

# np.diag(bl.omega)

ret_bl = bl.bl_returns()
# ret_bl

rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], index=["Prior", "Posterior", "Views"]).T
# rets_df

# rets_df.plot.bar(figsize=(12,8))
S_bl = bl.bl_cov()
# plotting.plot_covariance(S_bl)

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
# weights from BL using no ML insights
bl_weights = [w for w in weights.values()]

# test phase

ohlc_test = yf.download(tickers, start="2023-03-31", end="2023-05-01")
prices_test = ohlc_test["Adj Close"]

prices_returns = prices_test.pct_change().dropna()
prices_returns.to_excel('test_set_returns.xlsx')

passive_weights = [1/len(weights)]*len(weights)
passive_nav = passive_weights*(1 + prices_returns).cumprod()
bl_active_nav = bl_weights*(1 + prices_returns).cumprod()

passive_total_nav = passive_nav.sum(axis=1)
bl_active_total_nav = bl_active_nav.sum(axis=1)

# ML phase

ohlc_full = yf.download(tickers, start="2000-01-01", end="2023-04-30")
prices_full = ohlc_full["Adj Close"]
returns = np.log(1+prices_full.pct_change().dropna())

predictions = {}
for stock in tqdm(returns.columns):
    stock_s = returns[stock].copy()
    stock_df = stock_s.reset_index()

    # creating lags
    for j in range(1,11):
        stock_df['lag%s'%(j)] = stock_df[stock].shift(j)

    # ground truth
    stock_df['class_label'] = (stock_df[stock]>=0) + 0

    # remove nas
    stock_df.dropna(inplace=True)

    # X input columns
    Xcols = [c for c in stock_df if c.startswith('lag')]

    begDate = pd.Timestamp('2023-03-31')
    endDate = pd.Timestamp('2023-04-30')

    data_pre = stock_df[stock_df.Date <= begDate]
    data_pos = stock_df[(stock_df.Date > begDate) & (stock_df.Date <= endDate)]

    # mean and sigma of log returns
    avg = data_pre[stock].mean()
    std = data_pre[stock].std()

    X_train = np.array(data_pre[Xcols])
    Y_train = np.array(data_pre['class_label'])

    X_test = np.array(data_pos[Xcols])
    Y_test = np.array(data_pos['class_label'])

    with np.errstate(divide='ignore', invalid='ignore'):
        X_train = np.nan_to_num((X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0))
        X_test = np.nan_to_num((X_test - np.mean(X_train, axis=0))/np.std(X_train, axis=0))

    clf = MLPClassifier(random_state=0, max_iter=1000, solver='adam', hidden_layer_sizes=(41,41,41)).fit(X_train, Y_train)
    sp = list(clf.predict_proba(X_test)[:,1])

    returns_view = []
    for p in sp:
        z = st.norm.ppf(p)
        r = avg + z*std
        returns_view.append(r)

    if stock+'_pred' not in predictions:
        predictions[stock+'_pred'] = []
        predictions[stock+'_view'] = []
    predictions[stock+'_pred'].append(sp)
    predictions[stock+'_view'].append(returns_view)
# --- end of ML phase

# generating views
final_weights = []
for x,d in enumerate(data_pos.Date):
    pass
    viewdict = {}
    for tick in tickers:
        pass
        tick_view = predictions[tick+'_view'][0][x]
        if tick not in viewdict:
            viewdict[tick] = 0
        viewdict[tick] = tick_view

    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights = ef.clean_weights()

    bl_weights_df = pd.DataFrame(weights.values()).T
    bl_weights_df.columns = weights.keys()
    bl_weights_df.insert(loc=0, value=d, column='Date')
    final_weights.append(bl_weights_df)

fwd = pd.concat(final_weights, axis=0, ignore_index=True)
fwd.to_excel('ML_based_weights.xlsx', index=False)



print('DONE')