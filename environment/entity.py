import numpy as np
import pandas as pd
import yfinance as yf
from environment.cashflow import CashFlow
from scipy.stats.mstats import gmean
from datetime import datetime, timedelta

CURRENCIES = ["DKK", "SEK", "NOK", "EUR", "USD"]


class Balance:
    def __init__(self, currencies, home_ccy, balances, t, interest_frequency='quarterly'):
        self.currencies = currencies
        self.home_ccy = home_ccy
        self.values = pd.Series({ccy: balances[ccy] if balances is not None else 0 for ccy in currencies})
        self.num_trades = pd.Series()

        self.fx_costs = None
        self.fx_costs_local = None

        self.lending_costs = None
        self.lending_costs_local = None
        self.overdraft_costs = None
        self.overdraft_costs_local = None
        self.deposit_costs = None
        self.deposit_costs_local = None
        self.interest_costs = None
        self.interest_costs_local = None
        self.accrued_interest = pd.Series({ccy: 0 for ccy in self.currencies})
        self.next_interest_day = None
        self.interest_frequency = interest_frequency

        self.historical_values = None
        self.historical_values_local = None

        self.set_next_interest_day(t=t)

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
            self.interest_costs_local = pd.DataFrame(0, columns=self.currencies, index=[t])

            self.deposit_costs = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.lending_costs = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.overdraft_costs = pd.DataFrame(0, columns=self.currencies, index=[t])

            self.deposit_costs_local = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.lending_costs_local = pd.DataFrame(0, columns=self.currencies, index=[t])
            self.overdraft_costs_local = pd.DataFrame(0, columns=self.currencies, index=[t])
        else:
            self.interest_costs.loc[t] = 0
            self.interest_costs_local.loc[t] = 0

            self.deposit_costs.loc[t] = 0
            self.deposit_costs_local.loc[t] = 0

            self.lending_costs.loc[t] = 0
            self.lending_costs_local.loc[t] = 0

            self.overdraft_costs.loc[t] = 0
            self.overdraft_costs_local.loc[t] = 0

    def update_fx_cost(self, t, amount_from, ccy_from, ccy_to, margins, fx_rates):
        self.fx_costs.loc[t, ccy_from] += amount_from * margins.loc[ccy_from, ccy_to]
        self.fx_costs_local.loc[t, ccy_from] = self.fx_costs.loc[t, ccy_from] * fx_rates.loc[ccy_from, self.home_ccy]

    def set_next_interest_day(self, t):
        if self.interest_frequency == 'monthly':
            self.set_monthly(t=t)
        if self.interest_frequency == 'quarterly':
            self.set_quarterly(t=t)
        elif self.interest_frequency == "semi-annually":
            self.set_semi_annually(t=t)
        elif self.interest_frequency == "annually":
            self.set_annually(t=t)

    @staticmethod
    def get_next_business_day(t):
        if t.weekday() not in range(5):
            return t + timedelta(days=7 - t.weekday())
        else:
            return t

    def set_annually(self, t):
        annual_next_first_day = datetime(year=t.year+1, month=1, day=1)
        annual_next_first_day = self.get_next_business_day(t=annual_next_first_day)
        self.next_interest_day = annual_next_first_day

    def set_semi_annually(self, t):
        curr_half = (t.month - 1) // 6
        next_year = t.year if curr_half == 0 else t.year + 1
        half_next_first_day = datetime(year=next_year, month=6 * curr_half + 1, day=1)
        half_next_first_day = self.get_next_business_day(t=half_next_first_day)
        self.next_interest_day = half_next_first_day

    def set_quarterly(self, t):
        curr_quarter = t.month // 3 + 1
        next_month = ((3 * curr_quarter) % 12) + 1
        next_year = t.year if curr_quarter < 4 else t.year + 1
        q_next_first_day = datetime(year=next_year, month=next_month, day=1)
        q_next_first_day = self.get_next_business_day(t=q_next_first_day)
        self.next_interest_day = q_next_first_day

    def set_monthly(self, t):
        curr_month = t.month
        next_month = (curr_month + 1 % 12) + 1
        next_year = t.year if curr_month < 12 else t.year + 1
        m_next_first_day = datetime(year=next_year, month=next_month, day=1)
        m_next_first_day = self.get_next_business_day(t=m_next_first_day)
        return m_next_first_day

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

    def pay_interest(self, t):
        if t == self.next_interest_day:
            self.values -= self.accrued_interest
            self.accrued_interest = 0
            self.set_next_interest_day(t)

    def accrue_interest(self, t, deposit_rates, deposit_limits, lending_rates, lending_limits, overdraft_rates, fx_rates):
        deposit_costs = self.accrue_deposit_costs(deposit_rates=deposit_rates, deposit_limits=deposit_limits, fx_rates=fx_rates, t=t)
        lending_costs = self.accrue_lending_costs(lending_rates=lending_rates, lending_limits=lending_limits, fx_rates=fx_rates, t=t)
        overdraft_costs = self.accrue_overdraft_costs(overdraft_rates=overdraft_rates, lending_limits=lending_limits, fx_rates=fx_rates, t=t)
        accured_interest_today = deposit_costs + lending_costs + overdraft_costs
        self.accrued_interest -= accured_interest_today
        self.interest_costs.loc[t, :] = -accured_interest_today
        self.interest_costs_local.loc[t, :] = self.interest_costs.loc[t, :] * fx_rates.loc[:, self.home_ccy]

    def accrue_overdraft_costs(self, overdraft_rates, lending_limits, fx_rates, t):
        overdraft_rates = pd.Series(overdraft_rates)
        lending_limits = pd.Series(lending_limits)

        overdraft_costs = self.values[self.values < lending_limits] * overdraft_rates / 360
        overdraft_costs.fillna(0, inplace=True)
        self.overdraft_costs.loc[t] = overdraft_costs
        self.overdraft_costs_local.loc[t] = self.overdraft_costs.loc[t] * fx_rates.loc[:, self.home_ccy]
        return self.overdraft_costs.loc[t]

    def accrue_lending_costs(self, lending_rates, lending_limits, fx_rates, t):
        lending_rates = pd.Series(lending_rates)
        lending_limits = pd.Series(lending_limits)
        lending_cost = np.maximum(self.values[self.values < 0], lending_limits) * (lending_rates / 360)
        self.lending_costs.loc[t] = self.lending_costs.loc[t].add(lending_cost, fill_value=0)
        self.lending_costs_local.loc[t] = self.lending_costs.loc[t].multiply(fx_rates.loc[:, self.home_ccy])
        return self.lending_costs.loc[t]

    def accrue_deposit_costs(self, deposit_rates, deposit_limits, fx_rates, t):
        for ccy in self.currencies:
            ccy_deposit_rates = deposit_rates[ccy]
            ccy_deposit_limits = deposit_limits[ccy]
            balance = self.values.loc[ccy]
            accum_limit = 0
            for limit_prev, limit_next, rate in zip(ccy_deposit_limits[:-1], ccy_deposit_limits[1:], ccy_deposit_rates):
                if balance <= limit_prev:
                    break

                diff = limit_next - limit_prev
                accrue_amount = np.min((diff, balance - accum_limit))

                amount_interest = accrue_amount * rate / 360
                self.deposit_costs.loc[t, ccy] += amount_interest
                accum_limit += diff

                if balance <= limit_next:
                    break

            if balance > ccy_deposit_limits[-1]:
                rate = ccy_deposit_rates[-1]
                accrue_amount = balance - accum_limit
                amount_interest = accrue_amount * rate / 360
                self.deposit_costs.loc[t, ccy] += amount_interest
                self.deposit_costs_local.loc[t, ccy] = self.deposit_costs.loc[t, ccy] * fx_rates.loc[ccy, self.home_ccy]

        return self.deposit_costs.loc[t]

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
    def __init__(self, company_id, start_date, end_date, home_currency, trading_currencies,
                 fx_fees, interest_rates, cashflow, balance_t0=None):
        self.id = company_id
        self.start_date = start_date
        self.end_date = end_date
        self.start_balance = balance_t0

        self.fees = fx_fees
        self.cf = cashflow
        self.interest_rates = interest_rates

        self.domestic_ccy = home_currency
        self.ccy = trading_currencies

    def get_deposit_rates_range(self, t1, t2, idx=0):
        days = [t1 + timedelta(days=day) for day in range((t2 - t1).days + 1)]
        rates = pd.DataFrame(columns=self.ccy, index=days)
        for day in days:
            deposit_rates = self.get_deposit_rates(t=day)
            deposit_rate = {ccy: vals[idx] for ccy, vals in deposit_rates.items()}
            rates.loc[day, :] = deposit_rate
        return rates

    def get_deposit_rates(self, t):
        try:
            return self.interest_rates.get_rates(t=t, rate='deposit')
        except KeyError:
            return None

    def get_deposit_limits(self, t):
        try:
            return self.interest_rates.get_limits(t=t, limit="deposit")
        except KeyError:
            return None

    def get_lending_rates_range(self, t1, t2):
        days = [t1 + timedelta(days=day) for day in range((t2 - t1).days + 1)]
        rates = pd.DataFrame(columns=self.ccy, index=days)
        for day in days:
            rates.loc[day, :] = self.get_lending_rates(t=day)
        return rates

    def get_lending_rates(self, t):
        try:
            return self.interest_rates.get_rates(t=t, rate='lending')
        except KeyError:
            return None

    def get_lending_limits(self, t):
        try:
            return self.interest_rates.get_limits(t=t, limit="lending")
        except KeyError:
            return None

    def get_overdraft_rates_range(self, t1, t2):
        days = [t1 + timedelta(days=day) for day in range((t2 - t1).days + 1)]
        rates = pd.DataFrame(columns=self.ccy, index=days)
        for day in days:
            rates.loc[day, :] = self.get_overdraft_rates(t=day)
        return rates

    def get_overdraft_rates(self, t):
        try:
            return self.interest_rates.get_rates(t=t, rate='overdraft')
        except KeyError:
            return None

    def get_fees(self, t):
        try:
            return self.fees.get_fees(t=t)
        except KeyError:
            return None

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
        cf = CashFlow.generate_random(t1=t1, t2=t2, fx_rates=fx_rates, currencies=currencies, home_ccy=home_currency)
        return cls(company_id=company_id, start_date=t1, end_date=t2, home_currency=home_currency, trading_currencies=currencies,
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
            deposit_rates = self.rates[t_new][ccy]['deposit_rate'][0]
            new_lending_rates = deposit_rates + skew
            self.rates[t_new][ccy]['lending_rate'] = new_lending_rates

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
            if 'deposit_rate' not in self.rates[t_new][ccy]:
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
                    self.update_credit_limit(t_new=t_new, ccy=ccy, update=update_credit_limit)

                    update_deposit_levels = True if np.random.rand() < 0.025 else False
                    self.update_deposit_levels(t_new=t_new, ccy=ccy, update=update_deposit_levels)

                    update_lending_limit = True if np.random.rand() < 0.025 else False
                    self.update_lending_limit(t_new=t_new, ccy=ccy, update=update_lending_limit)

                    update_deposit_rate = True if np.random.rand() < 0.05 else False
                    self.update_deposit_rate(t_new=t_new, ccy=ccy, update=update_deposit_rate)

                    update_lending_rate = True if np.random.rand() < 0.05 else False
                    self.update_lending_rate(t_new=t_new, ccy=ccy, update=update_lending_rate)

                    update_overdraft_rate = True if np.random.rand() < 0.05 else False
                    self.update_overdraft_rate(t_new=t_new, ccy=ccy, update=update_overdraft_rate)

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


class FXMargins:
    def __init__(self, fees, day, currencies):
        self.t1 = day
        self.t = day
        self.currencies = currencies
        self.fees = fees

    def get_fees(self, t):
        return self.fees[t]

    def update_margin(self, fees, ccy):
        margin_change = 1 + np.random.normal(loc=0, scale=0.10, size=len(self.currencies))
        fees.loc[ccy, :] *= margin_change
        fees.loc[:, ccy] *= margin_change
        return fees

    def step(self, end_date=None):
        end_date = self.t + timedelta(days=1) if end_date is None else end_date

        while self.t < end_date:
            t_new = self.t + timedelta(days=1)
            fees_t_new = self.fees[self.t]
            if t_new.weekday() < 5:
                for ccy in self.currencies:
                    fees_t_new = self.update_margin(fees_t_new, ccy=ccy) if np.random.rand() < 0.025 else fees_t_new
                self.fees[t_new] = fees_t_new
            else:
                self.fees[t_new] = self.fees[self.t]
            self.t = t_new

    @classmethod
    def generate_random(cls, t1, t2, currencies):

        fees = {}
        fees_t0 = pd.DataFrame(
            np.random.uniform(low=0.005, high=0.05, size=(len(currencies), len(currencies))),
            columns=currencies,
            index=currencies)
        np.fill_diagonal(fees_t0.values, 0)
        fees[t1] = fees_t0

        inst = cls(fees=fees, day=t1, currencies=currencies)
        inst.step(end_date=t2)

        return inst


class FXRates:
    def __init__(self, fx_rates, fx_matrix, currencies, t1, t2):
        self.currencies = currencies
        self.rates = fx_rates
        self.fx_matrix = fx_matrix
        self.t = t1
        self.t_last = t2

        self.cov = None
        self.mean_vol = None
        self.mean = pd.Series(0, index=self.rates.columns)

    def add_new_day(self, new_rates):
        self.rates = self.rates.append(new_rates)

    def get_rates_today(self):
        return self.fx_matrix[self.t]

    def get_latest_row(self):
        return self.rates.loc[self.t_last]

    def get_latest_rates(self):
        return self.fx_matrix[self.t_last]

    def step(self):
        self.t += timedelta(days=1)

    def get_rates(self, t):
        try:
            return self.fx_matrix[t]
        except KeyError:
            return None

    def set_cov(self, cov):
        self.cov = cov

    def set_mean_vol(self, mean_vol):
        self.mean_vol = mean_vol

    def create_random_generator(self):
        returns = self.rates.diff()

        cov_mat = returns.cov()
        self.set_cov(cov=cov_mat)

        mean_vol = returns.rolling(window=30).mean().std()
        self.set_mean_vol(mean_vol=mean_vol)

    def generate_new(self, end_date=None):
        t = self.t_last + timedelta(days=1)
        t2 = t + timedelta(days=1) if end_date is None else end_date
        new_rates = self.get_latest_row()
        fx_matrices = self.fx_matrix

        while t < t2:
            mean = self.mean + np.random.normal(loc=0, scale=self.mean_vol)
            new_rates += np.random.multivariate_normal(mean=mean, cov=self.cov)
            new_rates.name = t
            self.add_new_day(new_rates=new_rates)
            FXRates.create_fx_matrix(close_rates_day=new_rates, day=t, fx_rates=fx_matrices)
            t += timedelta(days=1)

    @classmethod
    def generate_random(cls, t1, t2):
        tickers = ["{}{}=X".format(ccy1, ccy2) for ccy1 in CURRENCIES for ccy2 in CURRENCIES if ccy2 != ccy1]
        # historic_rates = yf.download(tickers=tickers, start=t1 + timedelta(days=1), end=t2 + timedelta(days=1))
        # close_rates = historic_rates['Close']
        # close_rates.to_csv('close_rates.csv', index=True)
        close_rates = pd.read_csv('close_rates.csv', index_col=0)
        close_rates.index = pd.to_datetime(close_rates.index)
        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)]

        for day in days:
            try:
                _ = close_rates.loc[day, :]
            except KeyError:
                close_rates.loc[day, :] = None

        close_rates.sort_index(inplace=True)
        close_rates.interpolate(inplace=True)
        close_rates.fillna(method='backfill', axis=0, inplace=True)

        fx_rates = cls.create_daily_fx_matrices(close_rates, days)
        inst = cls(fx_rates=close_rates, fx_matrix=fx_rates, currencies=CURRENCIES, t1=t1, t2=t2)
        inst.create_random_generator()
        return inst

    @classmethod
    def create_daily_fx_matrices(cls, close_rates, days):
        fx_rates = {}
        for day in days:
            close_rates_day = close_rates.loc[day]
            cls.create_fx_matrix(close_rates_day, day, fx_rates)
        return fx_rates

    @classmethod
    def create_fx_matrix(cls, close_rates_day, day, fx_rates):
        fx_rate_day = pd.DataFrame(columns=CURRENCIES, index=CURRENCIES)
        for ccy1 in CURRENCIES:
            for ccy2 in CURRENCIES:
                if ccy1 != ccy2:
                    fx_rate_day.loc[ccy1, ccy2] = close_rates_day.loc[ccy1 + ccy2 + "=X"]
                else:
                    fx_rate_day.loc[ccy1, ccy2] = 1
        fx_rates[day] = fx_rate_day
