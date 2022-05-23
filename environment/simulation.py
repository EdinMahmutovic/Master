from environment.entity import Company
from environment.entity import FXRates
from environment.agent import NoHandler, AutoFX
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import seaborn as sns
sns.set_theme()


class Simulation:
    def __init__(self, start_date, end_date, companies=(), num_companies=None, fx_rates=None):
        self.start_date = start_date if start_date.weekday() in range(5) \
            else start_date + timedelta(days=7 - start_date.weekday())

        self.end_date = end_date if end_date.weekday() in range(5) \
            else end_date - timedelta(days=end_date.weekday() - 4)

        self.num_companies = len(companies) if companies else num_companies
        self.fx_rates = fx_rates if fx_rates is not None else FXRates.generate_random(t1=start_date, t2=end_date)
        # self.fx_rates.generate_new(end_date=datetime(year=2025, month=1, day=1))

        self.companies = companies if companies \
            else [Company.generate_new(t1=self.start_date,
                                       t2=self.end_date,
                                       fx_rates=self.fx_rates.fx_matrix, company_id=company_id)
                  for company_id in range(self.num_companies)]

        self.models = [NoHandler(t=self.start_date, companies=self.companies),
                       AutoFX(t=self.start_date, companies=self.companies)]
        self.days = [self.start_date + timedelta(days=day) for day in range((self.end_date - self.start_date).days + 1)]

        for day in self.days:
            [model.update_balance(company, company.get_cashflow(t=day))
             for company in self.companies
             for model in self.models]

            close_fx_rates = self.fx_rates.get_rates(t=day)
            [model.rebalance(t=day, company=company, fx_rates=close_fx_rates)
             for company in self.companies
             for model in self.models]
