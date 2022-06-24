import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import numpy as np


class Scoreboard:
    def __init__(self, env):
        self.env = env

    def save(self, t1, t2):
        balances = {(model, company): model.get_historical_balances(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        num_trades = {(model, company): model.get_trades_per_day(company, t1=t1, t2=t2)
                      for model in self.env.models
                      for company in self.env.companies}

        fx_costs = {(model, company): model.get_fx_costs(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        interest_costs = {(model, company): model.get_interest_costs(company, local=True, t1=t1, t2=t2)
                          for model in self.env.models
                          for company in self.env.companies}

        deposit_costs = {(model, company): model.get_deposit_costs(company, local=True, t1=t1, t2=t2)
                         for model in self.env.models
                         for company in self.env.companies}

        lending_costs = {(model, company): model.get_lending_costs(company, local=True, t1=t1, t2=t2)
                         for model in self.env.models
                         for company in self.env.companies}

        overdraft_costs = {(model, company): model.get_overdraft_costs(company, local=True, t1=t1, t2=t2)
                           for model in self.env.models
                           for company in self.env.companies}

    def evaluate(self, t1, t2):

        # t1 = t1 if t1.weekday() in range(5) else t1 + timedelta(days=7 - t1.weekday())
        # t2 = t2 if t2.weekday() in range(5) else t2 - timedelta(days=t2.weekday() - 4)

        balances = {(model, company): model.get_historical_balances(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        num_trades = {(model, company): model.get_trades_per_day(company, t1=t1, t2=t2)
                      for model in self.env.models
                      for company in self.env.companies}

        # pd.DataFrame({(model, company): num_trades.sum() for (model, company), num_trades in num_trades.items()})
        fx_costs = {(model, company): model.get_fx_costs(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        interest_costs = {(model, company): model.get_interest_costs(company, local=True, t1=t1, t2=t2)
                          for model in self.env.models
                          for company in self.env.companies}

        fx_costs_autofx = None
        for model in self.env.models:
            figs, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 15), sharex='row')
            for company in self.env.companies:
                balances[model, company].plot(ax=axs[0, 0], label=model.id)
                axs[0, 0].set_title('Balances Company {}'.format(company.id))
                axs[0, 0].set_xlabel('Date')
                axs[0, 0].set_ylabel('Amount [DKK]')

                total_costs = fx_costs[model, company].cumsum(axis=0) + interest_costs[model, company].cumsum(axis=0)
                total_costs.plot(ax=axs[0, 1], label=model.id)
                axs[0, 1].set_title('Total Costs Company {}'.format(company.id))
                axs[0, 1].set_xlabel('Date')
                axs[0, 1].set_ylabel('Amount [DKK]')

                fx_costs[model, company].cumsum(axis=0).plot(ax=axs[1, 0], label=model.id)
                axs[1, 0].set_title('FX Costs Company {}'.format(company.id))
                axs[1, 0].set_xlabel('Date')
                axs[1, 0].set_ylabel('Amount [DKK]')

                interest_costs[model, company].cumsum(axis=0).plot(ax=axs[1, 1], label=model.id)
                axs[1, 1].set_title('Interest Costs Company {}'.format(company.id))
                axs[1, 1].set_xlabel('Date')
                axs[1, 1].set_ylabel('Amount [DKK]')

                lending_rates = company.get_lending_rates_range(t1=t1, t2=t2)
                lending_rates.plot(ax=axs[2, 0], label=model.id)
                axs[2, 0].legend()
                axs[2, 0].set_title('Lending Rates Company {}'.format(company.id))
                axs[2, 0].set_xlabel('Date')
                axs[2, 0].set_ylabel('Rate')

                deposit_rates = company.get_deposit_rates_range(t1=t1, t2=t2)
                deposit_rates.plot(ax=axs[2, 1], label=model.id)
                axs[2, 1].legend()
                axs[2, 1].set_title('Deposit Rate Company {}'.format(company.id))
                axs[2, 1].set_xlabel('Date')
                axs[2, 1].set_ylabel('Rate')

                print(f"Company {company.id}, Model {model.id}, total cost {total_costs.values.sum()}")

            plt.suptitle(model.id, fontsize=40)
            plt.show()
