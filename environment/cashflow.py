import numpy as np
from datetime import timedelta
import pandas as pd


class CashFlow:
    def __init__(self, t1, t2, currencies, home_ccy):
        self.start = t1
        self.end = t2
        self.t = t1
        self.currencies = currencies
        self.home_ccy = home_ccy

        self.baseline_rate = None

        self.num_customers = 0
        self.customer_ccy = []
        self.customer_income_per_transaction_ccy = []
        self.customer_num_transactions = []

        self.num_suppliers = 0
        self.supplier_ccy = []
        self.supplier_outgoing_per_transaction_ccy = []
        self.supplier_num_transactions = []

        days = [t1 + timedelta(days=days) for days in range((t2 - t1).days + 1)]
        self.data = pd.DataFrame(0, columns=currencies, index=days)
        self.data_local = pd.DataFrame(0, columns=currencies, index=days)

    def set_baseline_rate(self, fx_rates):
        self.baseline_rate = fx_rates

    def set_customer_currency(self, customer_currencies):
        self.customer_ccy = customer_currencies

    def set_supplier_currency(self, supplier_currencies):
        self.supplier_ccy = supplier_currencies

    def set_num_customers(self, num_customers):
        self.num_customers = num_customers

    def set_num_suppliers(self, num_suppliers):
        self.num_suppliers = num_suppliers

    def set_customer_income(self, customer_income):
        self.customer_income_per_transaction_ccy = customer_income

    def set_supplier_outgoing(self, supplier_outgoing):
        self.supplier_outgoing_per_transaction_ccy = supplier_outgoing

    def set_customer_transactions_per_year(self, annual_transactions):
        self.customer_num_transactions = annual_transactions

    def set_supplier_transactions_per_year(self, annual_transactions):
        self.supplier_num_transactions = annual_transactions

    def get_cashflow_period(self, t1, t2, local=False):
        if local:
            return self.data_local.loc[t1:t2, :]
        else:
            return self.data.loc[t1:t2, :]

    def get_cashflow(self, t, local=False):
        if local:
            return self.data_local.loc[t, :]
        else:
            return self.data.loc[t, :]

    @classmethod
    def generate_random(cls, t1, t2, fx_rates, currencies, home_ccy):
        base_ccy = cls.select_random_home_ccy(currencies)
        inst = cls(t1=t1, t2=t2, currencies=currencies, home_ccy=home_ccy)

        fx_rate_t0 = cls.get_baseline_fx_rates(fx_rates, t1)
        inst.set_baseline_rate(fx_rate_t0)

        inst.generate_random_supplier_info(currencies=currencies)
        inst.generate_random_customer_info(currencies=currencies)

        customer_income_share_pct, mean_income_annual = cls.generate_random_company_income(
            num_customers=inst.num_customers
        )

        mean_expense_annual, supplier_outgoing_share_pct = cls.generate_random_company_expenses(
            mean_income_annual=mean_income_annual,
            num_suppliers=inst.num_suppliers
        )

        mean_income_customer_annual = cls.compute_income_per_customer(customer_income_share_pct, mean_income_annual)
        mean_outgoing_supplier_annual = cls.compute_expense_per_supplier(mean_expense_annual,
                                                                         supplier_outgoing_share_pct)

        ccy_rates_customer = CashFlow.cashflow_ccy_conversion(base_ccy=base_ccy,
                                                              partner_ccy=inst.customer_ccy,
                                                              fx_rate_t0=fx_rate_t0)
        ccy_rates_supplier = CashFlow.cashflow_ccy_conversion(base_ccy=base_ccy,
                                                              partner_ccy=inst.supplier_ccy,
                                                              fx_rate_t0=fx_rate_t0)

        customer_income_per_transaction_local = CashFlow.compute_cashflow_per_transaction(
            partner_num_transactions=inst.customer_num_transactions,
            mean_cashflow_partner_annual=mean_income_customer_annual
        )
        supplier_outgoing_per_transaction_local = CashFlow.compute_cashflow_per_transaction(
            partner_num_transactions=inst.supplier_num_transactions,
            mean_cashflow_partner_annual=mean_outgoing_supplier_annual
        )

        customer_income_per_transaction_ccy = cls.compute_cashflow_in_ccy(
            ccy_rates=ccy_rates_customer,
            cashflow_per_transaction_local=customer_income_per_transaction_local)
        inst.set_customer_income(customer_income=customer_income_per_transaction_ccy)

        supplier_outgoing_per_transaction_ccy = cls.compute_cashflow_in_ccy(
            ccy_rates=ccy_rates_supplier,
            cashflow_per_transaction_local=supplier_outgoing_per_transaction_local)
        inst.set_supplier_outgoing(supplier_outgoing=supplier_outgoing_per_transaction_ccy)

        inst.step(fx_rates=fx_rates, end_date=t2)

        return inst

    def step(self, fx_rates, end_date=None):
        end_date = self.t + timedelta(days=1) if end_date is None else end_date

        while self.t < end_date:

            if self.t.weekday() < 5:
                self.generate_supplier_expenses(fx_rates)
                self.generate_customer_incomes(fx_rates)
            else:
                self.data.loc[self.t, :] = 0
                self.data_local.loc[self.t, :] = 0

            self.t = self.t + timedelta(days=1)

    def generate_customer_incomes(self, fx_rates):
        customer_iterator = zip(self.customer_ccy,
                                self.customer_num_transactions,
                                self.customer_income_per_transaction_ccy
                                )
        for ccy, num_trans, income in customer_iterator:
            if self.to_generate_cashflow(num_trans):
                self.sample_cashflow(ccy=ccy, fx_rates=fx_rates, avg_cashflow=income)

    def generate_supplier_expenses(self, fx_rates):
        supplier_iterator = zip(self.supplier_ccy,
                                self.supplier_num_transactions,
                                self.supplier_outgoing_per_transaction_ccy
                                )
        for ccy, num_trans, outgoing in supplier_iterator:
            if self.to_generate_cashflow(num_trans):
                self.sample_cashflow(ccy=ccy, fx_rates=fx_rates, avg_cashflow=-outgoing)

    @staticmethod
    def to_generate_cashflow(num_trans):
        return np.random.uniform(low=0, high=1) < num_trans / 360

    def sample_cashflow(self, ccy, fx_rates, avg_cashflow):
        self.data.loc[self.t, ccy] += np.random.normal(loc=avg_cashflow, scale=np.abs(avg_cashflow * 0.1))
        self.data_local.loc[self.t, ccy] = self.data.loc[self.t, ccy] * fx_rates[self.t].loc[ccy, self.home_ccy]

    @classmethod
    def compute_cashflow_in_ccy(cls, ccy_rates, cashflow_per_transaction_local):
        cashflow_per_transaction_ccy = cashflow_per_transaction_local * ccy_rates
        return cashflow_per_transaction_ccy

    @staticmethod
    def compute_cashflow_per_transaction(partner_num_transactions, mean_cashflow_partner_annual):
        return mean_cashflow_partner_annual / partner_num_transactions

    @staticmethod
    def cashflow_ccy_conversion(base_ccy, partner_ccy, fx_rate_t0):
        return [fx_rate_t0.loc[base_ccy, ccy] for ccy in partner_ccy]

    @staticmethod
    def compute_expense_per_supplier(mean_expense_annual, supplier_outgoing_share_pct):
        mean_outgoing_supplier_annual = mean_expense_annual * supplier_outgoing_share_pct
        return mean_outgoing_supplier_annual

    @staticmethod
    def compute_income_per_customer(customer_income_share_pct, mean_income_annual):
        mean_income_customer_annual = mean_income_annual * customer_income_share_pct
        return mean_income_customer_annual

    @staticmethod
    def generate_random_company_expenses(mean_income_annual, num_suppliers):
        mean_expense_annual = mean_income_annual * np.random.uniform(low=0.95, high=1.05)
        supplier_outgoing_share = np.random.uniform(low=1, high=100, size=num_suppliers)
        supplier_outgoing_share_pct = supplier_outgoing_share / np.sum(supplier_outgoing_share)
        return mean_expense_annual, supplier_outgoing_share_pct

    @staticmethod
    def generate_random_company_income(num_customers):
        mean_income_annual = np.random.uniform(low=10000, high=10 ** 9)
        customer_income_share = np.random.uniform(low=1, high=100, size=num_customers)
        customer_income_share_pct = customer_income_share / np.sum(customer_income_share)
        return customer_income_share_pct, mean_income_annual

    def generate_random_customer_info(self, currencies):
        num_customers = np.random.randint(low=1, high=10)
        self.set_num_customers(num_customers=num_customers)

        customer_num_transactions = np.random.randint(low=1, high=10, size=self.num_customers)
        self.set_customer_transactions_per_year(annual_transactions=customer_num_transactions)

        customer_ccy = np.random.choice(currencies, size=self.num_customers, replace=True)
        self.set_customer_currency(customer_currencies=customer_ccy)

    def generate_random_supplier_info(self, currencies):
        num_suppliers = np.random.randint(low=1, high=50)
        self.set_num_suppliers(num_suppliers=num_suppliers)

        supplier_num_transactions = np.random.randint(low=1, high=40, size=self.num_suppliers)
        self.set_supplier_transactions_per_year(annual_transactions=supplier_num_transactions)

        supplier_ccy = np.random.choice(currencies, size=self.num_suppliers, replace=True)
        self.set_supplier_currency(supplier_currencies=supplier_ccy)

    @staticmethod
    def get_baseline_fx_rates(fx_rates, t1):
        fx_rate_t0 = fx_rates[t1]
        return fx_rate_t0

    @staticmethod
    def select_random_home_ccy(currencies):
        base_ccy = currencies[0]
        return base_ccy
