import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from environment.cashflow import CashFlow

CURRENCIES = ["DKK", "SEK", "NOK", "EUR", "USD"]


class Balance:
    def __init__(self, currencies, home_ccy, balances):
        self.currencies = currencies
        self.home_ccy = home_ccy
        self.values = pd.Series({ccy: balances[ccy] if balances is not None else 0 for ccy in currencies})
        self.num_trades = pd.Series()
        self.fx_costs = None
        self.fx_costs_local = None
        self.interest_costs = None
        self.interest_costs_local = None
        self.historical_values = None
        self.historical_values_local = None

    def update_row(self, fx_rates, home_ccy):
        if self.historical_values is None:
            self.historical_values = self.values.to_frame().transpose()
            self.historical_values_local = (
                self.values.multiply(fx_rates.loc[self.currencies, home_ccy])).to_frame().transpose()
        else:
            self.historical_values = self.historical_values.append(self.values)
            local_values = self.values.multiply(fx_rates.loc[self.currencies, home_ccy]) if fx_rates is not None \
                else self.historical_values_local.iloc[-1, :].copy()

            local_values.name = self.values.name
            self.historical_values_local = self.historical_values_local.append(local_values)

    def make_transaction(self, from_ccy, to_ccy, amount_from, amount_to, margins, fx_rates, t):
        if amount_from > 0:
            self.num_trades.loc[t] += 1

        self.update_fx_cost(t=t, amount_from=amount_from, ccy_from=from_ccy, ccy_to=to_ccy,
                            margins=margins, fx_rates=fx_rates)
        self.withdraw(ccy=from_ccy, amount=amount_from)
        self.deposit(ccy=to_ccy, amount=amount_to)

    def init_day(self, t):
        self.num_trades = self.num_trades.append(pd.Series({t: 0}))

        if self.fx_costs is None:
            self.fx_costs = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.fx_costs_local = self.fx_costs.copy()
        else:
            self.fx_costs.loc[t] = 0
            self.fx_costs_local.loc[t] = 0

        if self.interest_costs is None:
            self.interest_costs = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.interest_costs_local = self.interest_costs.copy()
        else:
            self.interest_costs.loc[t] = 0
            self.interest_costs_local.loc[t] = 0

    def update_fx_cost(self, t, amount_from, ccy_from, ccy_to, margins, fx_rates):
        self.fx_costs.loc[t, ccy_from] += amount_from * margins.loc[ccy_from, ccy_to]
        self.fx_costs_local.loc[t, ccy_from] = self.fx_costs.loc[t, ccy_from] * fx_rates.loc[ccy_from, self.home_ccy]

    def get_fx_costs(self, local):
        if local:
            return self.fx_costs_local
        else:
            return self.fx_costs

    def get_interest_costs(self, local):
        if local:
            return self.interest_costs_local
        else:
            return self.interest_costs

    def pay_interest(self, t, deposit_rates, lending_rates, fx_rates):

        deposit_costs = self.values[self.values >= 0].multiply(deposit_rates)
        deposit_costs = deposit_costs[deposit_costs.notnull()]

        lending_costs = self.values[self.values < 0].multiply(lending_rates)
        lending_costs = lending_costs[lending_costs.notnull()]

        self.interest_costs.loc[t] = self.interest_costs.loc[t].add(deposit_costs, fill_value=0)
        self.interest_costs.loc[t] = self.interest_costs.loc[t].add(lending_costs, fill_value=0)

        self.interest_costs_local.loc[t] = self.interest_costs.loc[t].multiply(
            fx_rates.loc[self.currencies, self.home_ccy])
        self.values = self.values.add(self.interest_costs.loc[t])

    def deposit(self, ccy, amount):
        self.values[ccy] += amount

    def withdraw(self, ccy, amount):
        self.values[ccy] -= amount

    def update_balance(self, ccy, amount):
        self.values[ccy] -= amount

    def add(self, cashflow):
        self.values = self.values.add(cashflow) if cashflow is not None else self.values

    def get(self):
        return self.values

    def get_trades(self):
        return self.num_trades

    def get_historical(self, local):
        if local:
            return self.historical_values_local
        else:
            return self.historical_values


class Company:
    def __init__(self, id, start_date, end_date, home_currency, trading_currencies,
                 fx_fees, interest_rates, cashflow, balance_t0=None):
        self.id = id
        self.start_date = start_date
        self.end_date = end_date
        self.start_balance = balance_t0

        self.fees = fx_fees
        self.cf = cashflow
        self.interest_rates = interest_rates

        self.domestic_ccy = home_currency
        self.ccy = trading_currencies

    def get_deposit_rates(self, t):
        return self.interest_rates.get_rates(t=t, rate='deposit') / 360

    def get_lending_rates(self, t):
        return self.interest_rates.get_rates(t=t, rate='lending') / 360

    def get_overdraft_rates(self, t):
        return self.interest_rates.get_rates(t=t, rate='overdraft') / 360

    # @todo make fee data for everyday
    def get_fees(self, t):
        try:
            weekday = t.weekday()
            t = t if weekday < 5 else t - timedelta(days=weekday - 4)
            return self.fees.get_fees(t=t)
        except KeyError:
            return None

    # @todo make cashflow data for everyday
    def get_cashflow(self, t):
        try:
            return self.cf.get_cashflow(t=t)
        except KeyError:
            return None

    @classmethod
    def generate_new(cls, t1, t2, fx_rates, company_id=None):
        currencies = np.random.choice(CURRENCIES, size=np.random.randint(low=3, high=len(CURRENCIES)), replace=False)
        home_currency = np.random.choice(currencies)

        interest_rates = InterestRates.generate_random(t1=t1, t2=t2, currencies=currencies, home_ccy=home_currency)
        fx_fees = FXMargins.generate_random(t1=t1, t2=t2, currencies=currencies)
        cf = CashFlow.generate_random(t1=t1, t2=t2, fx_rates=fx_rates, ccys=currencies)
        return cls(id=company_id, start_date=t1, end_date=t2, home_currency=home_currency, trading_currencies=currencies,
                   fx_fees=fx_fees, interest_rates=interest_rates, cashflow=cf)


class InterestRates:
    def __init__(self, currencies, home_ccy, rates, day):
        self.t1 = day
        self.t = day
        self.currencies = currencies
        self.home_ccy = home_ccy
        self.rates = rates

    def get_deposit_levels(self, t, ccy=None):
        if ccy is None:
            deposit_limits = {ccy: self.rates[t][ccy]['deposit_limits'] for ccy in self.currencies}
            return deposit_limits
        else:
            return self.rates[t][ccy]['deposit_limits']

    def get_limits(self, t, limit, ccy=None):
        assert limit in ['deposit', 'lending', 'credit'], "pass valid limit type"
        if ccy is None:
            deposit_rates = {ccy: self.rates[t][ccy][f'{limit}_limits'] for ccy in self.currencies}
            return deposit_rates
        else:
            return self.rates[t][ccy][f'{limit}_limits']

    def get_rates(self, t, rate, ccy=None):
        assert rate in ['deposit', 'lending', 'overdraft'], "pass valid rate type"
        if ccy is None:
            deposit_rates = {ccy: self.rates[t][ccy][f'{rate}_rate'] for ccy in self.currencies}
            return deposit_rates
        else:
            return self.rates[t][ccy][f'{rate}_rate']

    def update_credit_limit(self, t_new, ccy, update=False):
        if update:
            self.rates[t_new][ccy]['credit_limits'] = 0 if ccy != self.home_ccy \
                else np.random.randint(low=-10 ** 7, high=-10 ** 5)
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='credit_limits')

    def update_deposit_levels(self, t_new, ccy, update=False):
        if update:
            self.modify_num_levels(ccy, t_new) if np.random.rand() > 0.5 \
                else self.update_deposit_amount_levels(ccy=ccy, t_new=t_new)
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='deposit_limits')

    def init_deposit_levels(self, t_new, ccy):
        num_deposit_levels = np.random.randint(low=0, high=3)
        deposit_levels = np.random.randint(low=1, high=10 ** 7, size=num_deposit_levels)
        deposit_limits = np.append(np.array([0]), deposit_levels)
        deposit_limits = np.sort(deposit_limits)
        deposit_limits = np.append(deposit_limits, deposit_limits[-1] + 0.01)
        self.rates[t_new][ccy]["deposit_limits"] = deposit_limits

    def modify_num_levels(self, ccy, t_new):
        self.decrease_num_deposit_levels(ccy, t_new) if np.random.rand() < 0.5 \
            else self.increase_num_deposit_levels(ccy=ccy, t_new=t_new)

    def decrease_num_deposit_levels(self, ccy, t_new):
        if len(self.rates[self.t][ccy]["deposit_limits"]) < 3:
            return self.increase_num_deposit_levels(ccy=ccy, t_new=t_new)
        deposit_limits = self.rates[self.t][ccy]["deposit_limits"][:-2]
        last_level = deposit_limits[-1]
        deposit_limits = np.insert(deposit_limits, deposit_limits.size, last_level+0.01)
        self.rates[t_new][ccy]["deposit_limits"] = deposit_limits

        deposit_rates = self.rates[self.t][ccy]["deposit_rate"]
        deposit_rates = deposit_rates[:-1]
        self.rates[t_new][ccy]["deposit_rate"] = deposit_rates

    def increase_num_deposit_levels(self, ccy, t_new):
        deposit_limits = self.rates[self.t][ccy]["deposit_limits"][:-1]
        new_limit = deposit_limits[-1] + np.random.randint(low=10 ** 4, high=10 ** 5)
        new_limits = np.array([new_limit, new_limit + 0.01])

        new_deposit_limits = np.append(deposit_limits, new_limits)
        self.rates[t_new][ccy]["deposit_limits"] = new_deposit_limits

        deposit_rates = self.rates[self.t][ccy]["deposit_rate"]
        last_rate = deposit_rates[-1] - np.random.uniform(low=0, high=0.005)
        deposit_rates = np.insert(deposit_rates, deposit_rates.size, last_rate)
        self.rates[t_new][ccy]["deposit_rate"] = deposit_rates

    def update_deposit_amount_levels(self, ccy, t_new):
        deposit_limits = self.rates[self.t][ccy]["deposit_limits"]
        deposit_limits[1:-1] += np.random.randint(low=-10 ** 4, high=10 ** 4, size=(len(deposit_limits) - 2))
        deposit_limits = np.sort(deposit_limits[:-1])
        last_limit = deposit_limits[-1]
        deposit_limits = np.insert(deposit_limits, deposit_limits.size, last_limit + 0.01)
        self.rates[t_new][ccy]["deposit_limits"] = deposit_limits

    def update_lending_limit(self, t_new, ccy, update=False):
        if update:
            lending_limit = self.rates[self.t][ccy]["lending_limits"]
            lending_limit += np.random.choice(np.linspace(start=-lending_limit, stop=10**6,
                                                          num=np.random.randint(low=2, high=20)))
            self.rates[t_new][ccy]["lending_limits"] = lending_limit
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='lending_limits')

    def init_lending_limit(self, t_new, ccy):
        lending_limit = np.random.choice(np.linspace(start=-0.1, stop=-10 ** 6, num=10))
        self.rates[t_new][ccy]["lending_limits"] = lending_limit

    def update_lending_rate(self, t_new, ccy, update=False):
        if update:
            lending_rate = self.rates[self.t][ccy]['lending_rate'] + np.random.normal(loc=0, scale=0.005)
            self.rates[t_new][ccy]['lending_rate'] = lending_rate
            self.check_lending_rate(ccy, t_new)
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='lending_rate')

    def check_lending_rate(self, ccy, t_new):
        lending_rate = self.rates[t_new][ccy]['lending_rate']
        if lending_rate < self.rates[t_new][ccy]['deposit_rate'][0]:
            skew = np.random.uniform(low=0, high=0.01)
            deposit_rates = self.rates[t_new][ccy]['deposit_rate']
            new_deposit_rates = deposit_rates + skew
            self.rates[t_new][ccy]['deposit_rate'] = new_deposit_rates

    def init_lending_rate(self, t_new, ccy):
        lending_rate = np.random.uniform(low=0.01, high=0.05)
        self.rates[t_new][ccy]['lending_rate'] = lending_rate
        self.check_lending_rate(ccy, t_new)

    def update_deposit_rate(self, t_new, ccy, update=False):
        if update:
            num_limits = len(self.rates[t_new][ccy]['deposit_limits'])
            random_step = np.random.normal(loc=0, scale=0.01, size=num_limits)
            try:
                deposit_rates = self.rates[t_new][ccy]['deposit_rate'] + random_step
            except KeyError:
                deposit_rates = self.rates[self.t][ccy]['deposit_rate'] + random_step
            deposit_rates = np.sort(deposit_rates)[::-1]
            self.rates[t_new][ccy]['deposit_rate'] = deposit_rates
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='deposit_rate')

    def init_deposit_rates(self, t_new, ccy):
        num_limits = len(self.rates[self.t][ccy]['deposit_limits'])
        deposit_rate0 = np.random.uniform(low=-0.005, high=0.05)
        deposit_rate_rest = deposit_rate0 - np.random.uniform(low=0.001, high=0.01, size=(num_limits - 1))
        deposit_rates = np.array([deposit_rate0] + [deposit_rate_rest[i] for i in range(num_limits - 1)])
        deposit_rates = np.sort(deposit_rates)[::-1]
        self.rates[t_new][ccy]['deposit_rate'] = deposit_rates

    def update_overdraft_rate(self, t_new, ccy, update=False):
        if update:
            overdraft_rate = self.rates[self.t][ccy]['overdraft_rate'] + np.random.normal(loc=0, scale=0.01)
            self.rates[t_new][ccy]['overdraft_rate'] = overdraft_rate
        else:
            self.maintain_previous_value(ccy=ccy, t_new=t_new, key='overdraft_rate')

    def init_overdraft_rate(self, t_new, ccy):
        overdraft_rate = np.random.uniform(low=0.07, high=0.10)
        self.rates[t_new][ccy]['overdraft_rate'] = overdraft_rate

    def maintain_previous_value(self, ccy, t_new, key):
        self.rates[t_new][ccy][key] = self.rates[self.t][ccy][key]

    def step(self, end_date=None):
        end_date = self.t + timedelta(days=1) if end_date is None else end_date

        while self.t < end_date:
            t_new = self.t + timedelta(days=1)
            self.rates[t_new] = {ccy: {} for ccy in self.currencies}
            if t_new.weekday() < 5:
                for ccy in self.currencies:

                    update_credit_limit = True if np.random.rand() < 0.0025 else False
                    self.update_credit_limit(t_new=t_new, ccy=ccy, update=True)

                    update_deposit_levels = True if np.random.rand() < 0.025 else False
                    self.update_deposit_levels(t_new=t_new, ccy=ccy, update=True)

                    update_lending_limit = True if np.random.rand() < 0.025 else False
                    self.update_lending_limit(t_new=t_new, ccy=ccy, update=True)

                    update_deposit_rate = True if np.random.rand() < 0.05 else False
                    self.update_deposit_rate(t_new=t_new, ccy=ccy, update=True)

                    update_lending_rate = True if np.random.rand() < 0.05 else False
                    self.update_lending_rate(t_new=t_new, ccy=ccy, update=True)

                    update_overdraft_rate = True if np.random.rand() < 0.05 else False
                    self.update_overdraft_rate(t_new=t_new, ccy=ccy, update=True)

            else:
                self.rates[t_new] = self.rates[self.t]
            self.t = t_new

    @classmethod
    def generate_random(cls, t1, t2, currencies, home_ccy):

        rates = {t1: {ccy: {} for ccy in currencies}}
        inst = cls(currencies=currencies, home_ccy=home_ccy, rates=rates, day=t1)
        for ccy in currencies:
            inst.update_credit_limit(t_new=t1, ccy=ccy, update=True)
            inst.init_deposit_levels(t_new=t1, ccy=ccy)
            inst.init_lending_limit(t_new=t1, ccy=ccy)
            inst.init_deposit_rates(t_new=t1, ccy=ccy)
            inst.init_lending_rate(t_new=t1, ccy=ccy)
            inst.init_overdraft_rate(t_new=t1, ccy=ccy)

        inst.step(end_date=t2)
        return inst


class Rates:
    def __init__(self, rates, rate_type):
        self.rates = rates
        self.rate_type = rate_type

    def get_rates(self, t=None):
        if t is not None:
            return self.rates[t]
        else:
            return self.rates

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
