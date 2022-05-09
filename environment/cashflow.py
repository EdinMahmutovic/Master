import numpy as np
from datetime import timedelta
import pandas as pd


class CashFlow:
    def __init__(self, start_date, end_date, daily_cashflows):
        self.start = start_date
        self.end = end_date
        self.currencies = daily_cashflows.columns.tolist()
        self.data = daily_cashflows

    def get_cashflow(self, t):
        return self.data.loc[t]

    @classmethod
    def generate_random(cls, t1, t2, fx_rates, ccys):
        base_ccy = ccys[0]
        fx_rate_t0 = fx_rates[t1]

        num_suppliers = np.random.randint(low=1, high=50)
        supplier_num_transactions = np.random.randint(low=1, high=50, size=num_suppliers)
        supplier_ccy = np.random.choice(ccys, size=num_suppliers, replace=True)

        num_customers = np.random.randint(low=1, high=50)
        customer_num_transactions = np.random.randint(low=1, high=50, size=num_customers)
        customer_ccy = np.random.choice(ccys, size=num_customers, replace=True)

        mean_income_annual = np.random.uniform(low=10000, high=10 ** 9)
        customer_income_share = np.random.uniform(low=1, high=100, size=num_customers)
        customer_income_share_pct = customer_income_share / np.sum(customer_income_share)

        mean_expense_annual = mean_income_annual
        supplier_outgoing_share = np.random.uniform(low=1, high=100, size=num_suppliers)
        supplier_outgoing_share_pct = supplier_outgoing_share / np.sum(supplier_outgoing_share)

        mean_income_customer_annual = mean_income_annual * customer_income_share_pct
        mean_outgoing_supplier_annual = mean_expense_annual * supplier_outgoing_share_pct

        ccy_convertions_customer = [fx_rate_t0.loc[base_ccy, ccy] for ccy in customer_ccy]
        ccy_convertions_supplier = [fx_rate_t0.loc[base_ccy, ccy] for ccy in supplier_ccy]

        customer_income_per_transaction = mean_income_customer_annual / customer_num_transactions
        supplier_outgoing_per_transaction = mean_outgoing_supplier_annual / supplier_num_transactions

        customer_income_per_transaction_ccy = customer_income_per_transaction * ccy_convertions_customer
        supplier_outgoing_per_transaction_ccy = supplier_outgoing_per_transaction * ccy_convertions_supplier

        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)
                if (t1 + timedelta(days=days)).weekday() in [0, 1, 2, 3, 4]]

        cashflow_df = pd.DataFrame(0, columns=ccys, index=days)
        cashflow_local_df = pd.DataFrame(0, columns=ccys, index=days)

        for day in days:
            for ccy, num_trans, outgoing in zip(supplier_ccy,
                                                supplier_num_transactions,
                                                supplier_outgoing_per_transaction_ccy
                                                ):
                if np.random.uniform(low=0, high=1) < num_trans / 360:
                    cashflow_df.loc[day, ccy] -= np.random.normal(loc=outgoing, scale=outgoing * 0.1)
                    cashflow_local_df.loc[day, ccy] = cashflow_df.loc[day, ccy] * fx_rates[day].loc[ccy, "DKK"]

            for ccy, num_trans, income in zip(customer_ccy,
                                              customer_num_transactions,
                                              customer_income_per_transaction_ccy
                                              ):
                if np.random.uniform(low=0, high=1) < num_trans / 360:
                    cashflow_df.loc[day, ccy] += np.random.normal(loc=income, scale=income * 0.1)
                    cashflow_local_df.loc[day, ccy] = cashflow_df.loc[day, ccy] * fx_rates[day].loc[ccy, "DKK"]

        return cls(start_date=t1, end_date=t2, daily_cashflows=cashflow_df)
