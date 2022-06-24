from environment.entity import Balance
import numpy as np
import pandas as pd
from mip import Model, xsum, maximize, BINARY, MINIMIZE
from datetime import timedelta


class LiquidityHandler:
    def __init__(self, name, t, companies=None, interest_frequency="quarterly"):
        self.id = name
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

    def get_deposit_costs(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_deposit_costs(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_deposit_costs(local=local)

    def get_lending_costs(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_lending_costs(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_lending_costs(local=local)

    def get_overdraft_costs(self, company, local=False, t1=None, t2=None):
        if (t1 is not None) & (t2 is not None):
            return self.balances[company].get_overdraft_costs(local=local).loc[t1:t2, :]
        else:
            return self.balances[company].get_overdraft_costs(local=local)

    def update_balance(self, company, cashflow):
        if cashflow is not None:
            self.balances[company].add(cashflow)


class NoHandler(LiquidityHandler):
    def __init__(self, name, t, companies=None):
        super().__init__(name=name, t=t, companies=companies)

    def rebalance(self, t, company, fx_rates):
        rates = fx_rates.get_rates(t=t)
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t

        deposit_rates = company.get_deposit_rates(t=t)
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

        self.balances[company].update_row(fx_rates=rates, home_ccy=company.domestic_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=rates)
        self.balances[company].pay_interest(t=t)


class AutoFX(LiquidityHandler):
    def __init__(self, name, t, companies=None):
        super().__init__(name=name, t=t, companies=companies)

    def reset_balances(self, t, from_ccy, company, balances, fx_rates, fx_margins):
        for to_ccy in company.ccy:
            if (to_ccy == company.domestic_ccy) or (to_ccy == from_ccy):
                continue

            missing_cash = -min(0, balances[to_ccy])
            amount2transfer_from = missing_cash / (
                    fx_rates.loc[from_ccy, to_ccy] * (1 - fx_margins.loc[from_ccy, to_ccy]))
            amount2transfer_from = max(0, min(balances[from_ccy], amount2transfer_from))

            self.balances[company].make_transaction(from_ccy=from_ccy, to_ccy=to_ccy,
                                                    amount_from=amount2transfer_from,
                                                    margins=fx_margins, fx_rates=fx_rates, t=t)

    def rebalance(self, t, company, fx_rates):
        rates = fx_rates.get_rates(t=t)
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t
        home_ccy = company.domestic_ccy

        if t.weekday() != 4:
            self.balances[company].update_row(fx_rates=rates, home_ccy=home_ccy)
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

            self.reset_balances(t=t, from_ccy=ccy1, company=company, balances=balances, fx_rates=rates, fx_margins=fees)

            surplus_cash_from = max(0, balances[ccy1])
            self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=home_ccy,
                                                    amount_from=surplus_cash_from,
                                                    margins=fees, fx_rates=rates, t=t)

        self.reset_balances(t=t, from_ccy=home_ccy, company=company, balances=balances, fx_rates=rates, fx_margins=fees)
        self.balances[company].update_row(fx_rates=rates, home_ccy=home_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=rates)
        self.balances[company].pay_interest(t=t)


class HindsightLP(LiquidityHandler):
    def __init__(self, name, t, companies=None, horizon=1, num_days=364, risk_willingness=0):
        super().__init__(name=name, companies=companies, t=t)
        self.num_days = num_days
        self.id = "HindsightLP"
        self.risk_level = risk_willingness
        self.models = {company: Model(sense=MINIMIZE) for company in self.companies}
        self.horizon = horizon

    def rebalance(self, t, company, fx_rates):
        rates = fx_rates.get_rates(t=t)
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t
        home_ccy = company.domestic_ccy

        ccy_volatility = fx_rates.get_volatility(home_ccy=home_ccy, currencies=company.ccy, t=t).values
        deposit_rate0 = company.get_future_deposit_rates(t1=t, horizon=self.horizon) / 360
        lending_rate0 = company.get_future_lending_rates(t1=t, horizon=self.horizon) / 360
        overdraft_rate0 = company.get_future_overdraft_rates(t1=t, horizon=self.horizon) / 360
        lending_limit = company.get_lending_limits_period(t1=t, horizon=self.horizon)

        exchange_rates = fx_rates.get_rates_period(currencies=company.ccy, t1=t, horizon=self.horizon)
        fx_margins = company.get_future_fees(t1=t, horizon=self.horizon)
        future_cashflow = company.get_future_cashflow(t1=t, horizon=self.horizon)
        start_balance = balances.copy()

        solution = self.build_model(t=t,
                                    company=company,
                                    c=future_cashflow.values.copy(),
                                    r_borrow=deposit_rate0.values.copy(),
                                    r_lend=lending_rate0.values.copy(),
                                    r_ovr=overdraft_rate0.values.copy(),
                                    m=fx_margins.copy(),
                                    b0=start_balance.values.copy(),
                                    l=lending_limit.values.copy(),
                                    sigma=ccy_volatility.copy(),
                                    k=self.risk_level,
                                    fx_rates=exchange_rates.copy())
        print(solution)
        #exit()
        fees = company.get_fees(t=t)
        deposit_rates = company.get_deposit_rates(t=t)
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

        for ccy1 in company.ccy:
            for ccy2 in company.ccy:
                self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=ccy2,
                                                        amount_from=solution.loc[ccy1, ccy2],
                                                        margins=fees, fx_rates=rates, t=t)

        self.balances[company].update_row(fx_rates=rates, home_ccy=home_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=rates)
        self.balances[company].pay_interest(t=t)

    def build_model(self, t, company, c, r_borrow, r_lend, r_ovr, m, b0, l, sigma, k, fx_rates, M=10 ** 8):
        days = [t + timedelta(days=i) for i in range(self.horizon)]
        weekends = [False if day.weekday() in range(5) else True for day in days]

        k = np.array([k if ccy != company.domestic_ccy else 0 for ccy in company.ccy])
        idx_hc = np.where(k == 0)[0][0]
        num_ccy = len(company.ccy)

        b0 /= 10000
        c /= 10000
        l /= 10000

        model = Model(solver_name='GRB', sense=MINIMIZE)

        model.solver.set_int_param("IntegralityFocus", 1)
        model.solver.set_int_param("NumericFocus", 3)
        model.solver.set_dbl_param("IntFeasTol", 1e-9)
        model.solver.set_dbl_param("FeasibilityTol", 1e-9)

        y = [[model.add_var(name="RATE_TYPE", var_type=BINARY) for _ in range(self.horizon)]
             for _ in range(num_ccy)]
        o = [[model.add_var(name="OVERDRAFT_TYPE", var_type=BINARY) for _ in range(self.horizon)]
             for _ in range(num_ccy)]

        b_pos = [[model.add_var(name="POSITIVE_BALANCE", lb=0, ub=10 ** 6) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]
        b_neg = [[model.add_var(name="NEGATIVE_BALANCE", lb=-10 ** 6, ub=0) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]
        b_ovr = [[model.add_var(name="OVERDRAFT_BALANCE", lb=-10 ** 6, ub=0) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]

        x = [[[model.add_var(name="TRANSFER", lb=0, ub=10 ** 6) for _ in range(self.horizon)]
              for _ in range(num_ccy)]
             for _ in range(num_ccy)]

        # fx costs is defined as the transactions fees paid per transaction.
        fx_costs = xsum(x[i][j][t] * m[i, j, t] * fx_rates[i, idx_hc, t]
                        for i in range(num_ccy)
                        for j in range(num_ccy)
                        for t in range(self.horizon)
                        if i != j)

        # deposit costs is defined as the costs/gains of holding a positive balance in a currency.
        deposit_costs = xsum(-b_pos[i][t] * r_borrow[t, i] * fx_rates[i, idx_hc, t]
                             for i in range(num_ccy)
                             for t in range(self.horizon))

        # lending costs is defined as the costs/gains of borrowing money in a currency.
        lending_costs = xsum(-b_neg[i][t] * r_lend[t, i] * fx_rates[i, idx_hc, t]
                             for i in range(num_ccy)
                             for t in range(self.horizon))

        # overdraft costs is defined as the costs of exceeding the lending limit in a currency.
        overdraft_costs = xsum(-b_ovr[i][t] * r_ovr[t, i] * fx_rates[i, idx_hc, t]
                               for i in range(num_ccy)
                               for t in range(self.horizon))

        # interest cost is defined as the sum of deposit-, lending and overdraft costs.
        interest_costs = deposit_costs + lending_costs + overdraft_costs

        # currency risk is defined as the total exposure to a currency. Where the risk is defined
        # as the standard deviation of a period (180 days).
        ccy_risk = xsum(k[i] * (b_pos[i][t] - b_neg[i][t] - b_ovr[i][t]) * sigma[i] * fx_rates[i, idx_hc, t]
                        for i in range(len(company.ccy))
                        for t in range(self.horizon))

        # the objective is the sum of fx costs, interest costs and currency risk.
        model.objective = fx_costs + interest_costs + ccy_risk

        # for i in range(num_ccy):
        #     for t in range(self.horizon):
        #         model += x[i][i][t] == 0

        # for i in range(num_ccy):
        #     for t in range(self.horizon):
        #         model += y[i][t] == 0
        #         model += o[i][t] == 1

        for t, weekend in enumerate(weekends):
            if weekend:
                for i in range(num_ccy):
                    for j in range(num_ccy):
                        model += x[i][j][t] == 0

        for i in range(num_ccy):
            for t in range(self.horizon):

                received = xsum(x[j][i][0] * (1 - m[j, i, 0]) * fx_rates[j, i, 0] for j in range(num_ccy) if j != i)
                transferred = xsum(x[i][j][0] for j in range(num_ccy) if j != i)
                b_start = b0[i] + received - transferred

                b_t = b_start
                for t_bar in range(1, t + 1):
                    received = xsum(
                        x[j][i][t_bar] * (1 - m[j, i, t_bar]) * fx_rates[j, i, t_bar] for j in range(num_ccy) if j != i)
                    transferred = xsum(x[i][j][t_bar] for j in range(num_ccy) if j != i)
                    b_t = b_t + received - transferred + c[t, i]

                # received = xsum(x[j][i][t_bar] * (1 - m[j, i, t_bar]) * fx_rates[j, i, t_bar]
                #                 for j in range(num_ccy)
                #                 for t_bar in range(t + 1))
                # transferred = xsum(x[i][j][t_bar]
                #                    for j in range(num_ccy)
                #                    for t_bar in range(t + 1))
                # cashflow = np.sum(c[:(t + 1), i])
                # b_t = b0[i] + received + cashflow - transferred

                # Binary Indicator if B is greater than 0. y = 1 if b > 0 else 0
                model += b_t >= -M * (1 - y[i][t])
                model += b_t <= M * y[i][t]

                model += b_t >= l[t, i] - M * o[i][t]
                model += b_t <= l[t, i] + M * (1 - o[i][t])

                model += b_pos[i][t] <= M * y[i][t]
                model += b_pos[i][t] <= b_t + M * (1 - y[i][t])
                model += b_pos[i][t] >= b_t - M * (1 - y[i][t])

                model += b_neg[i][t] >= -M * (1 - y[i][t])
                model += b_neg[i][t] >= l[t, i] - M * y[i][t]
                model += b_neg[i][t] >= b_t - M * y[i][t]
                model += b_neg[i][t] <= l[t, i] + M * (1 - o[i][t])
                model += b_neg[i][t] <= b_t + M * o[i][t]

                model += b_ovr[i][t] >= -M * o[i][t]
                model += b_ovr[i][t] <= (b_t - l[t, i]) + M * (1 - o[i][t])
                model += b_ovr[i][t] >= (b_t - l[t, i]) - M * (1 - o[i][t])

        model.optimize(max_seconds=60)

        for i in range(num_ccy):
            for t in range(self.horizon):
                o_var = round(o[i][t].x, 4)
                y_var = round(y[i][t].x, 4)

                lim = round(l[t, i], 4)

                b_minus = round(b_neg[i][t].x, 4)
                b_plus = round(b_pos[i][t].x, 4)
                b_overdraft = round(b_ovr[i][t].x, 4)

                assert (o_var == 1 and b_minus <= lim) or (o_var == 0 and b_minus >= lim), f'overdraft error, o={o_var}, b_neg={b_minus}. limit={l[t][i]}'
                assert (o_var == 1 and b_overdraft <= 0) or (o_var == 0 and b_overdraft == 0), f"overdraft error, o={o_var}, b_ovr={b_overdraft}"
                assert (y_var == 1 and b_minus >= 0) or (y_var == 0 and b_minus <= 0), f'sign error minus, y={y_var}, b_neg={b_minus}'
                assert (y_var == 1 and b_plus >= 0) or (y_var == 0 and b_plus <= 0), f'sign error plus, y={y_var}, b_neg={b_plus}'
                assert (b_minus == 0 and b_plus >= 0) or (b_minus <= 0 and b_plus == 0), f"constraint error, b_neg={b_minus}, b_pos={b_pos}"

        solution = pd.DataFrame(0, columns=company.ccy, index=company.ccy)

        return solution


class ForecastLP(LiquidityHandler):
    def __init__(self, name, t, forecaster, companies=None, horizon=1, num_days=364, risk_willingness=0):
        super().__init__(name=name, companies=companies, t=t)
        self.num_days = num_days
        self.id = "ForecastLP"
        self.risk_level = risk_willingness
        self.models = {company: Model(sense=MINIMIZE) for company in self.companies}
        self.horizon = horizon
        self.model = forecaster

    def rebalance(self, t, company, fx_rates):
        rates = fx_rates.get_rates(t=t)
        balances = self.get_balance(company)
        self.balances[company].init_day(t=t)
        balances.name = t
        home_ccy = company.domestic_ccy

        ccy_volatility = fx_rates.get_volatility(home_ccy=home_ccy, currencies=company.ccy, t=t)
        cashflow = company.get_future_cashflow(t1=t, horizon=self.horizon)
        future_cashflow = self.model(cashflow)
        start_balance = balances.copy()

        exchange_rates = rates.loc[company.ccy, company.ccy]
        fx_margins = company.get_fees(t=t)
        deposit_rate0 = company.get_future_deposit_rates(t1=t, horizon=1).iloc[0, :] / 360
        lending_rate0 = company.get_future_lending_rates(t1=t, horizon=1).iloc[0, :] / 360
        overdraft_rate0 = company.get_future_overdraft_rates(t1=t, horizon=1).iloc[0, :] / 360
        lending_limit = company.get_lending_limits_period(t1=t, horizon=1).iloc[0, :]

        solution = self.build_model(t=t,
                                    company=company,
                                    c=future_cashflow.values.copy(),
                                    r_borrow=deposit_rate0.values.copy(),
                                    r_lend=lending_rate0.values.copy(),
                                    r_ovr=overdraft_rate0.values.copy(),
                                    m=fx_margins.values.copy(),
                                    b0=start_balance.values.copy(),
                                    l=lending_limit.values.copy(),
                                    sigma=ccy_volatility.values.copy(),
                                    k=self.risk_level,
                                    fx_rates=exchange_rates.values.copy())
        print(solution)
        fees = company.get_fees(t=t)
        deposit_rates = company.get_deposit_rates(t=t)
        deposit_limits = company.get_deposit_limits(t=t)
        lending_rates = company.get_lending_rates(t=t)
        lending_limits = company.get_lending_limits(t=t)
        overdraft_rates = company.get_overdraft_rates(t=t)

        for ccy1 in company.ccy:
            for ccy2 in company.ccy:
                self.balances[company].make_transaction(from_ccy=ccy1, to_ccy=ccy2,
                                                        amount_from=solution.loc[ccy1, ccy2],
                                                        margins=fees, fx_rates=rates, t=t)

        self.balances[company].update_row(fx_rates=rates, home_ccy=home_ccy)
        self.balances[company].accrue_interest(t=t,
                                               deposit_rates=deposit_rates,
                                               deposit_limits=deposit_limits,
                                               lending_rates=lending_rates,
                                               lending_limits=lending_limits,
                                               overdraft_rates=overdraft_rates,
                                               fx_rates=rates)
        self.balances[company].pay_interest(t=t)

    def build_model(self, t, company, c, r_borrow, r_lend, r_ovr, m, b0, l, sigma, k, fx_rates, M=10 ** 8):
        days = [t + timedelta(days=i) for i in range(self.horizon)]
        weekends = [False if day.weekday() in range(5) else True for day in days]

        k = np.array([k if ccy != company.domestic_ccy else 0 for ccy in company.ccy])
        idx_hc = np.where(k == 0)[0][0]
        num_ccy = len(company.ccy)

        b0 /= 10000
        c /= 10000
        l /= 10000

        model = Model(solver_name='GRB', sense=MINIMIZE)

        model.solver.set_int_param("IntegralityFocus", 1)
        model.solver.set_int_param("NumericFocus", 3)
        model.solver.set_dbl_param("IntFeasTol", 1e-9)
        model.solver.set_dbl_param("FeasibilityTol", 1e-9)

        y = [[model.add_var(name="RATE_TYPE", var_type=BINARY) for _ in range(self.horizon)]
             for _ in range(num_ccy)]
        o = [[model.add_var(name="OVERDRAFT_TYPE", var_type=BINARY) for _ in range(self.horizon)]
             for _ in range(num_ccy)]

        b_pos = [[model.add_var(name="POSITIVE_BALANCE", lb=0, ub=10 ** 6) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]
        b_neg = [[model.add_var(name="NEGATIVE_BALANCE", lb=-10 ** 6, ub=0) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]
        b_ovr = [[model.add_var(name="OVERDRAFT_BALANCE", lb=-10 ** 6, ub=0) for _ in range(self.horizon)]
                 for _ in range(num_ccy)]

        x = [[[model.add_var(name="TRANSFER", lb=0, ub=10 ** 6) for _ in range(self.horizon)]
              for _ in range(num_ccy)]
             for _ in range(num_ccy)]

        # fx costs is defined as the transactions fees paid per transaction.
        fx_costs = xsum(x[i][j][t] * m[i, j] * fx_rates[i, idx_hc]
                        for i in range(num_ccy)
                        for j in range(num_ccy)
                        for t in range(self.horizon)
                        if i != j)

        # deposit costs is defined as the costs/gains of holding a positive balance in a currency.
        deposit_costs = xsum(-b_pos[i][t] * r_borrow[i] * fx_rates[i, idx_hc]
                             for i in range(num_ccy)
                             for t in range(self.horizon))

        # lending costs is defined as the costs/gains of borrowing money in a currency.
        lending_costs = xsum(-b_neg[i][t] * r_lend[i] * fx_rates[i, idx_hc]
                             for i in range(num_ccy)
                             for t in range(self.horizon))

        # overdraft costs is defined as the costs of exceeding the lending limit in a currency.
        overdraft_costs = xsum(-b_ovr[i][t] * r_ovr[i] * fx_rates[i, idx_hc]
                               for i in range(num_ccy)
                               for t in range(self.horizon))

        # interest cost is defined as the sum of deposit-, lending and overdraft costs.
        interest_costs = deposit_costs + lending_costs + overdraft_costs

        # currency risk is defined as the total exposure to a currency. Where the risk is defined
        # as the standard deviation of a period (180 days).
        ccy_risk = xsum(k[i] * (b_pos[i][t] - b_neg[i][t] - b_ovr[i][t]) * sigma[i] * fx_rates[i, idx_hc]
                        for i in range(len(company.ccy))
                        for t in range(self.horizon))

        # the objective is the sum of fx costs, interest costs and currency risk.
        model.objective = fx_costs + interest_costs + ccy_risk

        # for i in range(num_ccy):
        #     for t in range(self.horizon):
        #         model += x[i][i][t] == 0

        # for i in range(num_ccy):
        #     for t in range(self.horizon):
        #         model += y[i][t] == 0
        #         model += o[i][t] == 1

        for t, weekend in enumerate(weekends):
            if weekend:
                for i in range(num_ccy):
                    for j in range(num_ccy):
                        model += x[i][j][t] == 0

        for i in range(num_ccy):
            for t in range(self.horizon):

                received = xsum(x[j][i][0] * (1 - m[j, i]) * fx_rates[j, i] for j in range(num_ccy) if j != i)
                transferred = xsum(x[i][j][0] for j in range(num_ccy) if j != i)
                b_start = b0[i] + received - transferred

                b_t = b_start
                for t_bar in range(1, t + 1):
                    received = xsum(
                        x[j][i][t_bar] * (1 - m[j, i]) * fx_rates[j, i] for j in range(num_ccy) if j != i)
                    transferred = xsum(x[i][j][t_bar] for j in range(num_ccy) if j != i)
                    b_t = b_t + received - transferred + c[t, i]

                # received = xsum(x[j][i][t_bar] * (1 - m[j, i, t_bar]) * fx_rates[j, i, t_bar]
                #                 for j in range(num_ccy)
                #                 for t_bar in range(t + 1))
                # transferred = xsum(x[i][j][t_bar]
                #                    for j in range(num_ccy)
                #                    for t_bar in range(t + 1))
                # cashflow = np.sum(c[:(t + 1), i])
                # b_t = b0[i] + received + cashflow - transferred

                # Binary Indicator if B is greater than 0. y = 1 if b > 0 else 0
                model += b_t >= -M * (1 - y[i][t])
                model += b_t <= M * y[i][t]

                model += b_t >= l[i] - M * o[i][t]
                model += b_t <= l[i] + M * (1 - o[i][t])

                model += b_pos[i][t] <= M * y[i][t]
                model += b_pos[i][t] <= b_t + M * (1 - y[i][t])
                model += b_pos[i][t] >= b_t - M * (1 - y[i][t])

                model += b_neg[i][t] >= -M * (1 - y[i][t])
                model += b_neg[i][t] >= l[i] - M * y[i][t]
                model += b_neg[i][t] >= b_t - M * y[i][t]
                model += b_neg[i][t] <= l[i] + M * (1 - o[i][t])
                model += b_neg[i][t] <= b_t + M * o[i][t]

                model += b_ovr[i][t] >= -M * o[i][t]
                model += b_ovr[i][t] <= (b_t - l[i]) + M * (1 - o[i][t])
                model += b_ovr[i][t] >= (b_t - l[i]) - M * (1 - o[i][t])

        model.optimize(max_seconds=10)

        for i in range(num_ccy):
            for t in range(self.horizon):
                o_var = round(o[i][t].x, 4)
                y_var = round(y[i][t].x, 4)

                lim = round(l[i], 4)

                b_minus = round(b_neg[i][t].x, 4)
                b_plus = round(b_pos[i][t].x, 4)
                b_overdraft = round(b_ovr[i][t].x, 4)

                assert (o_var == 1 and b_minus <= lim) or (o_var == 0 and b_minus >= lim), f'overdraft error, o={o_var}, b_neg={b_minus}. limit={l[t][i]}'
                assert (o_var == 1 and b_overdraft <= 0) or (o_var == 0 and b_overdraft == 0), f"overdraft error, o={o_var}, b_ovr={b_overdraft}"
                assert (y_var == 1 and b_minus >= 0) or (y_var == 0 and b_minus <= 0), f'sign error minus, y={y_var}, b_neg={b_minus}'
                assert (y_var == 1 and b_plus >= 0) or (y_var == 0 and b_plus <= 0), f'sign error plus, y={y_var}, b_neg={b_plus}'
                assert (b_minus == 0 and b_plus >= 0) or (b_minus <= 0 and b_plus == 0), f"constraint error, b_neg={b_minus}, b_pos={b_pos}"

        solution = pd.DataFrame(0, columns=company.ccy, index=company.ccy)

        return solution


class StochasticLP(LiquidityHandler):
    def __init__(self, regression_model, t, companies=None, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days
        self.regression = regression_model


class RLAgent(LiquidityHandler):
    def __init__(self, t, companies=None, num_days=364):
        super().__init__(companies=companies, t=t)
        self.num_days = num_days
