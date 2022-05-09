from environment.entity import Balance
import numpy as np
import pandas as pd


class LiquidityHandler:
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

    def get_historical_balances(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_historical(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_historical(local=local)

    def get_trades_per_day(self, company, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_trades().loc[t1:t2]
        else:
            return self.balances[company].get_trades()

    def get_total_trades(self, company, t1, t2):
        num_trades = self.balances[company].get_trades()
        return num_trades.loc[t1:t2].sum()

    def get_fx_costs(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_fx_costs(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_fx_costs(local=local)

    def get_interest_costs(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_interest_costs(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_interest_costs(local=local)

    def update_balance(self, company, cashflow):
        if cashflow is not None:
            self.balances[company].add(cashflow)


class NoHandler(LiquidityHandler):
    def __init__(self, companies=None):
        super().__init__(companies=companies)
        self.id = "NoHandler"

    def rebalance(self, t, company, fx_rates):
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t

        deposit_rates = company.get_deposit_rates(t=t)
        lending_rates = company.get_lending_rates(t=t)

        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=company.domestic_ccy)
        self.balances[company].pay_interest(t=t, deposit_rates=deposit_rates, lending_rates=lending_rates,
                                            fx_rates=fx_rates)


class AutoFX(LiquidityHandler):
    def __init__(self, companies=None):
        super().__init__(companies=companies)
        self.id = "AutoFX"

    def reset_balances(self, t, from_ccy, company, balances, fx_rates, fx_margins):
        for to_ccy in company.ccy:
            if (to_ccy == company.domestic_ccy) or (to_ccy == from_ccy):
                continue

            missing_cash = -min(0, balances[to_ccy])
            amount2transfer_from = missing_cash / (fx_rates.loc[from_ccy, to_ccy] * (1 - fx_margins.loc[from_ccy, to_ccy]))
            amount2transfer_from = max(0, min(balances[from_ccy], amount2transfer_from))

            amount2transfer_to = amount2transfer_from * (1 - fx_margins.loc[from_ccy, to_ccy]) * fx_rates.loc[from_ccy, to_ccy]

            self.balances[company].make_transaction(from_ccy=from_ccy, to_ccy=to_ccy,
                                                    amount_from=amount2transfer_from, amount_to=amount2transfer_to,
                                                    margins=fx_margins, fx_rates=fx_rates, t=t)

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

            self.reset_balances(t=t, from_ccy=ccy1, company=company, balances=balances, fx_rates=fx_rates, fx_margins=fees)

            surplus_cash_from = max(0, balances[ccy1])
            surplus_cash_to = surplus_cash_from * fx_rates.loc[ccy1, home_ccy] * (1 - fees.loc[ccy1, home_ccy])

            self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=home_ccy,
                                                    amount_from=surplus_cash_from, amount_to=surplus_cash_to,
                                                    margins=fees, fx_rates=fx_rates, t=t)

        self.reset_balances(t=t, from_ccy=home_ccy, company=company, balances=balances, fx_rates=fx_rates, fx_margins=fees)
        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=home_ccy)
        self.balances[company].pay_interest(t=t, deposit_rates=deposit_rates, lending_rates=lending_rates,
                                            fx_rates=fx_rates)


# @Todo implement hindsight LP which knows all future values for n days
class HindsightLP(LiquidityHandler):
    def __init__(self, companies=None, num_days=364):
        super().__init__(companies=companies)
        self.num_days = num_days
        self.id = "HindsightLP"


# @Todo implement forecast LP which predicts future cashflows for n days and assumes constant fees and interest rates.
class ForecastLP(LiquidityHandler):
    def __init__(self, companies=None, num_days=364):
        super().__init__(companies=companies)
        self.num_days = num_days


# @Todo implement forecast LP which predicts future cashflow distribution and assumes constant fees and interest rates.
class StochasticLP(LiquidityHandler):
    def __init__(self, companies=None, num_days=364):
        super().__init__(companies=companies)
        self.num_days = num_days


# @Todo implement reinforcement learning agent which minimizes the overall cost.
class DQN(LiquidityHandler):
    def __init__(self, companies=None, num_days=364):
        super().__init__(companies=companies)
        self.num_days = num_days
