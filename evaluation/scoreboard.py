import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd


class Scoreboard:
    def __init__(self, env):
        self.env = env

    def evaluate(self, t1, t2):

        t1 = t1 if t1.weekday() in range(5) else t1 + timedelta(days=7 - t1.weekday())
        t2 = t2 if t2.weekday() in range(5) else t2 - timedelta(days=t2.weekday() - 4)

        days = [day for day in self.env.days if t1 <= day <= t2]
        balances = {(model, company): model.get_historical_balances(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        num_trades = {(model, company): model.get_trades_per_day(company, t1=t1, t2=t2)
                      for model in self.env.models
                      for company in self.env.companies}

        total_trades = {(model, company): num_trades.sum() for (model, company), num_trades in num_trades.items()}

        fx_costs = {(model, company): model.get_fx_costs(company, local=True, t1=t1, t2=t2)
                    for model in self.env.models
                    for company in self.env.companies}

        interest_costs = {(model, company): model.get_interest_costs(company, local=True, t1=t1, t2=t2)
                          for model in self.env.models
                          for company in self.env.companies}

        for model in self.env.models:
            figs, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 15), sharex='row', sharey='row')
            for company in self.env.companies:
                balances[model, company].plot(ax=axs[0, 0], label=model.id)
                axs[0, 0].set_title('Balances Company {}'.format(company.id))
                axs[0, 0].set_xlabel('Date')
                axs[0, 0].set_ylabel('Amount [DKK]')

                total_costs = fx_costs[model, company].cumsum(axis=0) + interest_costs[model, company].cumsum(axis=0)
                axs[0, 1].plot(total_costs, label=model.id)
                axs[0, 1].set_title('Total Costs Company {}'.format(company.id))
                axs[0, 1].set_xlabel('Date')
                axs[0, 1].set_ylabel('Amount [DKK]')

                axs[1, 0].plot(fx_costs[model, company].cumsum(axis=0), label=model.id)
                #axs[1, 0].plot(fx_costs[model, company], label=model.id)
                axs[1, 0].set_title('FX Costs Company {}'.format(company.id))
                axs[1, 0].set_xlabel('Date')
                axs[1, 0].set_ylabel('Amount [DKK]')

                axs[1, 1].plot(interest_costs[model, company].cumsum(axis=0), label=model.id)
                #axs[1, 1].plot(interest_costs[model, company], label=model.id)
                axs[1, 1].set_title('Interest Costs Company {}'.format(company.id))
                axs[1, 1].set_xlabel('Date')
                axs[1, 1].set_ylabel('Amount [DKK]')

                pd.DataFrame.from_dict(company.lending_rates.get_rates()).T.loc[t1:t2, :].plot(ax=axs[2, 0], label=model.id)
                axs[2, 0].set_title('Lending Rates Company {}'.format(company.id))
                axs[2, 0].set_xlabel('Date')
                axs[2, 0].set_ylabel('Rate')

                pd.DataFrame.from_dict(company.deposit_rates.get_rates()).T.loc[t1:t2, :].plot(ax=axs[2, 1], label=model.id)
                axs[2, 1].set_title('Deposit Rate Company {}'.format(company.id))
                axs[2, 1].set_xlabel('Date')
                axs[2, 1].set_ylabel('Rate')

            plt.suptitle(model.id, fontsize=40)
            plt.show()
