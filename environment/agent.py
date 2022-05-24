from environment.entity import Balance
import numpy as np
import pandas as pd
from mip import Model, xsum, maximize, BINARY, MINIMIZE
from DQN import DQN


class LiquidityHandler:
    def __init__(self, t, companies=None, interest_frequency="quarterly"):
        companies = companies if companies is not None else set()
        self.companies = companies
        self.balances = {company: Balance(currencies=company.ccy,
                                          home_ccy=company.domestic_ccy,
                                          balances=company.start_balance,
                                          t=t,
                                          interest_frequency=interest_frequency)
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
    def __init__(self, t, companies=None):
        super().__init__(t=t, companies=companies)
        self.id = "NoHandler"

    def rebalance(self, t, company, fx_rates):
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t

        deposit_rates = company.get_deposit_rates(t=t)
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=company.domestic_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)


class AutoFX(LiquidityHandler):
    def __init__(self, t, companies=None):
        super().__init__(t=t, companies=companies)
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
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

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
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)


class HindsightLP(LiquidityHandler):
    def __init__(self, t, companies=None, horizon=28, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days
        self.id = "HindsightLP"
        self.models = {company: Model(sense=MINIMIZE) for company in self.companies}
        self.horizon = horizon
        self.build_model(horizon)

    def build_model(self, company, c, r_borrow, r_lend, m, b0, sigma, k, M=10**12):
        model = self.models[company]
        y = [[model.add_var(name="RATE_TYPE", var_type=BINARY) for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        r = [[model.add_var(name="RATE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        b = [[model.add_var(name="BALANCE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        x = [[model.add_var(name="TRANSFER", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]
        z = [[model.add_var(name="RECEIVE", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]

        for i in range(len(company.ccy)):
            for j in range(len(company.ccy)):
                for t in range(self.horizon):
                    model += z[t][j][i] == x[t][j][i] * m[t][j][i]

        for i in range(len(company.ccy)):
            model += b[i][0] + xsum(z[i][j][0] for j in range(len(company.ccy))) == b0

        for i in range(len(company.ccy)):
            for t in range(1, self.horizon):
                model += b[i][t] == b[i][t - 1] + c[i][t] + xsum(z[i][j][t] for j in range(len(company.ccy)))

        for i in range(len(company.ccy)):
            for t in range(self.horizon):
                model += b[i][t] >= M * (1 - y[i][t])
                model += b[i][t] <= M * y[i][t]

                model += r_borrow[i][t] - M * (1 - y[i][t]) <= r[i][t]
                model += r_borrow[i][t] + M * (1 - y[i][t]) >= r[i][t]

                model += r_lend[i][t] - M * y[i][t] <= r[i][t]
                model += r_lend[i][t] + M * y[i][t] >= r[i][t]

        fx_costs = xsum(x[i][j][t] * m[i][j][t]
                        for i in range(len(company.ccy))
                        for j in range(len(company.ccy))
                        for t in range(self.horizon))

        interest_costs = xsum(b[i][t] * r[i][t]
                              for i in range(len(company.ccy))
                              for t in range(self.horizon))

        ccy_risk = xsum(k[i] * b[i][t] * y[i][t] * sigma[i]
                        for i in range(len(company.ccy))
                        for t in range(self.horizon))

        model.objective = fx_costs + interest_costs + ccy_risk

    def solve(self, company):
        model = self.models[company]
        model.optimize()
        obj_val = model.objective_value
        x = model.var_by_name('TRANSFER')
        return obj_val, x.x

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
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

        for ccy1 in company.ccy:
            if ccy1 == home_ccy:
                continue

            self.reset_balances(t=t, from_ccy=ccy1, company=company, balances=balances, fx_rates=fx_rates,
                                fx_margins=fees)

            surplus_cash_from = max(0, balances[ccy1])
            surplus_cash_to = surplus_cash_from * fx_rates.loc[ccy1, home_ccy] * (1 - fees.loc[ccy1, home_ccy])

            self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=home_ccy,
                                                    amount_from=surplus_cash_from, amount_to=surplus_cash_to,
                                                    margins=fees, fx_rates=fx_rates, t=t)

        self.reset_balances(t=t, from_ccy=home_ccy, company=company, balances=balances, fx_rates=fx_rates,
                            fx_margins=fees)
        self.balances[company].update_row(fx_rates=fx_rates, home_ccy=home_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)


class ForecastLP(LiquidityHandler):
    def __init__(self, regression_model, t, companies=None, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days
        self.regression = regression_model

    def forcast(self, history):
        return self.regression(history)

    def build_model(self, company, c, r_borrow, r_lend, m, b0, sigma, k, M=10 ** 12):
        model = self.models[company]
        y = [[model.add_var(name="RATE_TYPE", var_type=BINARY) for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        r = [[model.add_var(name="RATE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        b = [[model.add_var(name="BALANCE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        x = [[model.add_var(name="TRANSFER", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]
        z = [[model.add_var(name="RECEIVE", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]

        for i in range(len(company.ccy)):
            for j in range(len(company.ccy)):
                for t in range(self.horizon):
                    model += z[t][j][i] == x[t][j][i] * m[t][j][i]

        for i in range(len(company.ccy)):
            model += b[i][0] + xsum(z[i][j][0] for j in range(len(company.ccy))) == b0

        for i in range(len(company.ccy)):
            for t in range(1, self.horizon):
                model += b[i][t] == b[i][t - 1] + c[i][t] + xsum(z[i][j][t] for j in range(len(company.ccy)))

        for i in range(len(company.ccy)):
            for t in range(self.horizon):
                model += b[i][t] >= M * (1 - y[i][t])
                model += b[i][t] <= M * y[i][t]

                model += r_borrow[i][t] - M * (1 - y[i][t]) <= r[i][t]
                model += r_borrow[i][t] + M * (1 - y[i][t]) >= r[i][t]

                model += r_lend[i][t] - M * y[i][t] <= r[i][t]
                model += r_lend[i][t] + M * y[i][t] >= r[i][t]

        fx_costs = xsum(x[i][j][t] * m[i][j][t]
                        for i in range(len(company.ccy))
                        for j in range(len(company.ccy))
                        for t in range(self.horizon))

        interest_costs = xsum(b[i][t] * r[i][t]
                              for i in range(len(company.ccy))
                              for t in range(self.horizon))

        ccy_risk = xsum(k[i] * b[i][t] * y[i][t] * sigma[i]
                        for i in range(len(company.ccy))
                        for t in range(self.horizon))

        model.objective = fx_costs + interest_costs + ccy_risk

    def solve(self, company):
        model = self.models[company]
        model.optimize()
        obj_val = model.objective_value
        x = model.var_by_name('TRANSFER')
        return obj_val, x.x

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
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

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
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)


class StochasticLP(LiquidityHandler):
    def __init__(self, regression_model, t, companies=None, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days

        self.regression = regression_model

    def forcast(self, history):
        return self.regression(history)

    def build_model(self, company, c, r_borrow, r_lend, m, b0, sigma, k, M=10 ** 12):
        model = self.models[company]
        y = [[model.add_var(name="RATE_TYPE", var_type=BINARY) for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        r = [[model.add_var(name="RATE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        b = [[model.add_var(name="BALANCE") for _ in range(len(company.ccy))]
             for _ in range(self.horizon)]
        x = [[model.add_var(name="TRANSFER", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]
        z = [[model.add_var(name="RECEIVE", lb=0)] for _ in range(len(company.ccy))
             for _ in range(len(company.ccy))
             for _ in range(self.horizon)]

        for i in range(len(company.ccy)):
            for j in range(len(company.ccy)):
                for t in range(self.horizon):
                    model += z[t][j][i] == x[t][j][i] * m[t][j][i]

        for i in range(len(company.ccy)):
            model += b[i][0] + xsum(z[i][j][0] for j in range(len(company.ccy))) == b0

        for i in range(len(company.ccy)):
            for t in range(1, self.horizon):
                model += b[i][t] == b[i][t - 1] + c[i][t] + xsum(z[i][j][t] for j in range(len(company.ccy)))

        for i in range(len(company.ccy)):
            for t in range(self.horizon):
                model += b[i][t] >= M * (1 - y[i][t])
                model += b[i][t] <= M * y[i][t]

                model += r_borrow[i][t] - M * (1 - y[i][t]) <= r[i][t]
                model += r_borrow[i][t] + M * (1 - y[i][t]) >= r[i][t]

                model += r_lend[i][t] - M * y[i][t] <= r[i][t]
                model += r_lend[i][t] + M * y[i][t] >= r[i][t]

        fx_costs = xsum(x[i][j][t] * m[i][j][t]
                        for i in range(len(company.ccy))
                        for j in range(len(company.ccy))
                        for t in range(self.horizon))

        interest_costs = xsum(b[i][t] * r[i][t]
                              for i in range(len(company.ccy))
                              for t in range(self.horizon))

        ccy_risk = xsum(k[i] * b[i][t] * y[i][t] * sigma[i]
                        for i in range(len(company.ccy))
                        for t in range(self.horizon))

        model.objective = fx_costs + interest_costs + ccy_risk

    def solve(self, company):
        model = self.models[company]
        model.optimize()
        obj_val = model.objective_value
        x = model.var_by_name('TRANSFER')
        return obj_val, x.x

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
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

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
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)


class RLAgent(LiquidityHandler):
    def __init__(self, t, companies=None, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days
        self.dqn = DQN(h=20, w=len(companies), outputs=companies.ccy)

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
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

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
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=fx_rates)
        self.balances[company].pay_interest(t=t)
