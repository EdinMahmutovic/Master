from environment.entity import Company
from environment.entity import FXRates
from environment.agent import NoHandler, AutoFX, HindsightLP, ForecastLP
from environment.forecast import testy as cf_model
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from environment.cashflow import CashFlow, CashFlowDataLoader
from torch.utils.data import DataLoader
import seaborn as sns
import pickle
import os
sns.set_theme()


class Simulation:
    def __init__(self, start_date, end_date, t1, t2, companies=(), num_companies=None, fx_rates=None):
        self.start_date = start_date if start_date.weekday() in range(5) \
            else start_date + timedelta(days=7 - start_date.weekday())

        self.end_date = end_date if end_date.weekday() in range(5) \
            else end_date - timedelta(days=end_date.weekday() - 4)

        self.num_companies = len(companies) if companies else num_companies
        self.fx_rates = fx_rates if fx_rates is not None else FXRates.generate_random(t1=t1, t2=t2)
        # self.fx_rates.generate_new(end_date=datetime(year=2025, month=1, day=1))

        # currencies = self.fx_rates.currencies
        # cf_data = CashFlowDataLoader(t1=t1, t2=t2, fx_rates=self.fx_rates.fx_matrix, currencies=currencies, n_series=100, write_new=True)
        # cf_dataloader = DataLoader(cf_data, batch_size=32)

        # self.companies = companies if companies \
        #     else [Company.generate_new(t1=t1,
        #                                t2=t2,
        #                                fx_rates=self.fx_rates.fx_matrix, company_id=company_id)
        #           for company_id in range(self.num_companies)]


        self.companies = [pickle.load(open("companies/" + file, 'rb')) for file in os.listdir("companies/")]

        exit()
        # self.models = [NoHandler(name="NoHandler", t=self.start_date, companies=self.companies),
        #               AutoFX(name="AutoFX", t=self.start_date, companies=self.companies)]
        self.models = [
            #AutoFX(name="AutoFX", t=self.start_date, companies=self.companies),
            HindsightLP(name="Clairvoyant28", t=self.start_date, companies=self.companies, horizon=14)
            #ForecastLP(name="ForecastLP7", t=self.start_date, forecaster=cf_model, companies=self.companies, horizon=7)
        ]

        self.days = [self.start_date + timedelta(days=day) for day in range((self.end_date - self.start_date).days + 1)]

        for day in self.days:
            [model.update_balance(company, company.get_cashflow(t=day))
             for company in self.companies
             for model in self.models]

            [model.rebalance(t=day, company=company, fx_rates=self.fx_rates)
             for company in self.companies
             for model in self.models]
