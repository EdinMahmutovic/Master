import numpy as np
import pandas as pd


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
            self.historical_values_local = (self.values.multiply(fx_rates.loc[:, home_ccy])).to_frame().transpose()
        else:
            self.historical_values = self.historical_values.append(self.values)
            local_values = self.values.multiply(fx_rates.loc[:, home_ccy]) if fx_rates is not None \
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

        self.interest_costs.loc[t] = self.interest_costs.loc[t].sub(deposit_costs, fill_value=0)
        self.interest_costs.loc[t] = self.interest_costs.loc[t].add(lending_costs, fill_value=0)

        self.interest_costs_local.loc[t] = self.interest_costs.loc[t].multiply(fx_rates.loc[:, self.home_ccy])
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


class NoHandler:
    def __init__(self, companies=None):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(currencies=company.ccy,
                                          home_ccy=company.domestic_ccy,
                                          balances=company.start_balance)
                         for company in companies}

    def add_company(self, company):
        self.companies.add(company)

    def get_historical_balances(self, company, local=False):
        return self.balances[company].get_historical(local=local)

    def get_balance(self, company):
        return self.balances[company].get()

    def get_trades_per_day(self, company):
        return self.balances[company].get_trades()

    def get_total_trades(self, company, t1, t2):
        num_trades = self.balances[company].get_trades()
        return num_trades.loc[t1:t2].sum()

    def get_fx_costs(self, company, local=False):
        return self.balances[company].get_fx_costs(local=local)

    def get_interest_costs(self, company, local=False):
        return self.balances[company].get_interest_costs(local=local)

    def get_total_fx_costs(self, company, t1, t2, local=False):
        fx_costs = self.balances[company].get_fx_costs(local=local)
        total_costs = fx_costs.loc[t1:t2, :].sum()
        return total_costs

    def update_balance(self, company, cashflow):
        if cashflow is not None:
            self.balances[company].add(cashflow)

    def rebalance(self, t, company, fx_rates):
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t

        deposit_rates = company.get_deposit_rates(t=t)
        lending_rates = company.get_lending_rates(t=t)

        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=company.domestic_ccy)
        self.balances[company].pay_interest(t=t, deposit_rates=deposit_rates, lending_rates=lending_rates,
                                            fx_rates=fx_rates)


class AutoFX:
    def __init__(self, companies=None):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(currencies=company.ccy,
                                          home_ccy=company.domestic_ccy,
                                          balances=company.start_balance)
                         for company in companies}

    def add_company(self, company):
        self.companies.add(company)

    def get_balance(self, company):
        return self.balances[company].get()

    def get_historical_balances(self, company, local=False):
        return self.balances[company].get_historical(local=local)

    def get_fx_costs(self, company, local=False):
        return self.balances[company].get_fx_costs(local=local)

    def get_interest_costs(self, company, local=False):
        return self.balances[company].get_interest_costs(local=local)

    def get_total_fx_costs(self, company, t1, t2, local=False):
        fx_costs = self.balances[company].get_fx_costs(local=local)
        total_costs = fx_costs.loc[t1:t2, :].sum()
        return total_costs

    def get_trades_per_day(self, company):
        return self.balances[company].get_trades()

    def get_total_trades(self, company, t1, t2):
        num_trades = self.balances[company].get_trades()
        return num_trades.loc[t1:t2].sum()

    def update_balance(self, company, cashflow):
        if cashflow is not None:
            self.balances[company].add(cashflow)

    def rebalance(self, t, company, fx_rates):
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t
        home_ccy = company.domestic_ccy

        if t.weekday() != 4:
            self.balances[company].update_row(fx_rates=fx_rates, home_ccy=home_ccy)
            return

        fees = company.get_fees(t=t)
        deposit_rates = company.get_deposit_rates(t=t)
        lending_rates = company.get_lending_rates(t=t)

        for ccy1 in company.ccy:
            if ccy1 == home_ccy:
                continue

            for ccy2 in company.ccy:
                if (ccy2 == home_ccy) or (ccy2 == ccy1):
                    continue

                missing_cash = -min(0, balances[ccy2])

                amount2transfer_from = missing_cash / (fx_rates.loc[ccy1, ccy2] * (1 - fees.loc[ccy1, ccy2]))
                amount2transfer_from = max(0, min(balances[ccy1], amount2transfer_from))

                amount2transfer_to = amount2transfer_from * (1 - fees.loc[ccy1, ccy2]) * fx_rates.loc[ccy1, ccy2]

                self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=ccy2,
                                                        amount_from=amount2transfer_from, amount_to=amount2transfer_to,
                                                        margins=fees, fx_rates=fx_rates, t=t)

            surplus_cash_from = max(0, balances[ccy1])
            surplus_cash_to = surplus_cash_from * fx_rates.loc[ccy1, home_ccy] * (1 - fees.loc[ccy1, home_ccy])

            self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=home_ccy,
                                                    amount_from=surplus_cash_from, amount_to=surplus_cash_to,
                                                    margins=fees, fx_rates=fx_rates, t=t)
        for ccy in company.ccy:
            if ccy == home_ccy:
                continue

            missing_cash = -min(0, balances[ccy])
            amount2transfer_from = missing_cash * (1 + fees.loc[home_ccy, ccy]) / (fx_rates.loc[home_ccy, ccy])
            amount2transfer_from = max(0, min(balances[home_ccy], amount2transfer_from))

            amount2transfer_to = amount2transfer_from * (1 - fees.loc[home_ccy, ccy]) * fx_rates.loc[home_ccy, ccy]

            self.balances[company].make_transaction(from_ccy=home_ccy, to_ccy=ccy,
                                                    amount_from=amount2transfer_from, amount_to=amount2transfer_to,
                                                    margins=fees, fx_rates=fx_rates, t=t)

        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=home_ccy)
        self.balances[company].pay_interest(t=t, deposit_rates=deposit_rates, lending_rates=lending_rates,
                                            fx_rates=fx_rates)


# @Todo implement hindsight LP which knows all future values for n days
class HindsightLP:
    def __init__(self, companies=None, num_days=364):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(company.ccy, company.start_balance) for company in companies}


# @Todo implement forecast LP which predicts future cashflows for n days and assumes constant fees and interest rates.
class ForecastLP:
    def __init__(self, companies=None, num_days=364):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(company.ccy, company.start_balance) for company in companies}


# @Todo implement forecast LP which predicts future cashflow distribution and assumes constant fees and interest rates.
class StochasticLP:
    def __init__(self, companies=None, num_days=364):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(company.ccy, company.start_balance) for company in companies}


# @Todo implement reinforcement learning agent which minimizes the overall cost.
class DQN:
    def __init__(self, companies=None, num_days=364):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(company.ccy, company.start_balance) for company in companies}
