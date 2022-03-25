import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from environment.cashflow import CashFlow

CURRENCIES = ["DKK", "SEK", "NOK", "EUR", "USD"]


class Company:
    def __init__(self, start_date, end_date, home_currency, trading_currencies,
                 fx_fees, deposit_rates, lending_rates, cashflow, balance_t0=None):
        self.start_date = start_date
        self.end_date = end_date
        self.start_balance = balance_t0

        self.fees = fx_fees
        self.deposit_rates = deposit_rates
        self.lending_rates = lending_rates
        self.cf = cashflow

        self.domestic_ccy = home_currency
        self.ccy = trading_currencies

    def get_lending_rates(self, t):
        try:
            weekday = t.weekday()
            t = t if weekday < 5 else t - timedelta(days=weekday - 4)
            return self.lending_rates.get_rates(t=t) / 360
        except KeyError:
            return None

    def get_deposit_rates(self, t):
        try:
            weekday = t.weekday()
            t = t if weekday < 5 else t - timedelta(days=weekday - 4)
            return self.deposit_rates.get_rates(t=t) / 360
        except KeyError:
            return None

    def get_fees(self, t):
        try:
            weekday = t.weekday()
            t = t if weekday < 5 else t - timedelta(days=weekday - 4)
            return self.fees.get_fees(t=t)
        except KeyError:
            return None

    def get_cashflow(self, t):
        try:
            return self.cf.get_cashflow(t=t)
        except KeyError:
            return None

    @classmethod
    def generate_new(cls, t1, t2, fx_rates):
        currencies = np.random.choice(CURRENCIES, size=np.random.randint(low=3, high=len(CURRENCIES)), replace=False)
        home_currency = np.random.choice(currencies)

        deposit_rates = Rates.generate_random(t1=t1, t2=t2, rate_type='DEPOSIT', currencies=currencies)
        lending_rates = Rates.generate_random(t1=t1, t2=t2, rate_type='LENDING', currencies=currencies)
        fx_fees = FXMargins.generate_random(t1=t1, t2=t2, currencies=currencies)
        cf = CashFlow.generate_random(t1=t1, t2=t2, fx_rates=fx_rates, ccys=currencies)
        return cls(start_date=t1, end_date=t2, home_currency=home_currency, trading_currencies=currencies,
                   fx_fees=fx_fees, deposit_rates=deposit_rates, lending_rates=lending_rates, cashflow=cf)


class Rates:
    def __init__(self, rates, rate_type):
        self.rates = rates
        self.rate_type = rate_type

    def get_rates(self, t):
        return self.rates[t]

    @classmethod
    def generate_random(cls, t1, t2, rate_type, currencies):
        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)
                if (t1 + timedelta(days=days)).weekday() in [0, 1, 2, 3, 4]]

        low_rate = -0.02 if rate_type == 'DEPOSIT' else 0.02
        high_rate = 0.04 if rate_type == 'DEPOSIT' else 0.06

        rates = {}
        rates_t0 = pd.Series(
            np.random.uniform(low=low_rate, high=high_rate, size=len(currencies)),
            index=currencies)
        rates[days[0]] = rates_t0

        for i, day in enumerate(days[1:]):
            new_rates = rates[days[i]].copy()
            for ccy in currencies:
                if np.random.rand() < 0.05:
                    new_rates[ccy] += np.random.normal(loc=0, scale=0.1)
                else:
                    new_rates[ccy] += np.random.normal(loc=0, scale=0.001)
            rates[day] = new_rates

        return cls(rates=rates, rate_type=rate_type)


class FXMargins:
    def __init__(self, fees):
        self.fees = fees

    def get_fees(self, t):
        return self.fees[t]

    @classmethod
    def generate_random(cls, t1, t2, currencies):
        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)
                if (t1 + timedelta(days=days)).weekday() in [0, 1, 2, 3, 4]]

        fees = {}
        fees_t0 = pd.DataFrame(
            np.random.uniform(low=0.005, high=0.05, size=(len(currencies), len(currencies))),
            columns=currencies,
            index=currencies)
        np.fill_diagonal(fees_t0.values, 0)
        fees[days[0]] = fees_t0

        for i, day in enumerate(days[1:]):
            new_fees = fees[days[i]].copy()
            for ccy in currencies:
                if np.random.rand() < 0.05:
                    new_fees[ccy] *= 1 + np.random.normal(loc=0, scale=0.10)
                else:
                    new_fees[ccy] *= 1 + np.random.normal(loc=0, scale=0.01)
            np.fill_diagonal(new_fees.values, 0)
            fees[day] = new_fees

        return cls(fees=fees)


class FXRates:
    def __init__(self, fx_rates, cov=None, corr=None, cross_volatility=None):
        self.rates = fx_rates
        self.cov = cov
        self.corr = corr
        self.volatility = cross_volatility

    def get_rates(self, t):
        try:
            weekday = t.weekday()
            t = t if weekday < 5 else t - timedelta(days=weekday - 4)
            return self.rates[t]
        except KeyError:
            return None

    @classmethod
    def generate_random(cls, t1, t2):
        tickers = ["{}{}=X".format(ccy1, ccy2) for ccy1 in CURRENCIES for ccy2 in CURRENCIES if ccy2 != ccy1]
        # historic_rates = yf.download(tickers=tickers, start=t1 + timedelta(days=1), end=t2 + timedelta(days=1))
        # close_rates = historic_rates['Close']
        # close_rates.to_csv('close_rates.csv', index=True)
        close_rates = pd.read_csv('close_rates.csv', index_col=0)
        close_rates.index = pd.to_datetime(close_rates.index)
        close_rates_returns = close_rates.pct_change()

        cov_mat = close_rates_returns.cov()
        corr_mat = close_rates_returns.corr()
        rolling_var = close_rates_returns.rolling(window=30).var().var()

        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)
                if (t1 + timedelta(days=days)).weekday() in [0, 1, 2, 3, 4]]

        fx_rates = {}
        error_dates = []
        for day in days:
            try:
                close_rates_day = close_rates.loc[day]
            except KeyError:
                error_dates.append(day)
                continue

            fx_rate_day = pd.DataFrame(columns=CURRENCIES, index=CURRENCIES)

            for ccy1 in CURRENCIES:
                for ccy2 in CURRENCIES:
                    if ccy1 != ccy2:
                        fx_rate_day.loc[ccy1, ccy2] = close_rates_day.loc[ccy1 + ccy2 + "=X"]
                    else:
                        fx_rate_day.loc[ccy1, ccy2] = 1

            fx_rates[day] = fx_rate_day

        for day in error_dates:
            idx = days.index(day)
            fx_rates[day] = fx_rates[days[idx - 1]] if idx > 0 else fx_rates[days[idx + 1]]

        return cls(fx_rates=fx_rates, cov=cov_mat, corr=corr_mat, cross_volatility=rolling_var)
