from environment.entity import Company
from environment.entity import FXRates
from environment.agent import NoHandler, AutoFX
import matplotlib.pyplot as plt
from datetime import timedelta
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

        self.companies = companies if companies \
            else [Company.generate_new(t1=self.start_date,
                                       t2=self.end_date,
                                       fx_rates=self.fx_rates.rates)
                  for _ in range(self.num_companies)]

        self.models = [NoHandler(companies=self.companies), AutoFX(companies=self.companies)]
        all_days = [self.start_date + timedelta(days=day) for day in range((self.end_date - self.start_date).days)]

        for day in all_days:
            [model.update_balance(company, company.get_cashflow(t=day))
             for company in self.companies
             for model in self.models]

            close_fx_rates = self.fx_rates.get_rates(t=day)
            [model.rebalance(t=day, company=company, fx_rates=close_fx_rates)
             for company in self.companies
             for model in self.models]

            #print("WEEKDAY:: {}".format(day.weekday()))
            #print(self.models[0].get_balance(self.companies[0]))
            #print(self.models[1].get_balance(self.companies[0]))
            #print("\n")

        lastyear = [day for day in all_days if day.year == 2021]
        balances0 = self.models[0].get_historical_balances(self.companies[0], local=True)
        balances1 = self.models[1].get_historical_balances(self.companies[0], local=True)

        num_trades0 = self.models[0].get_trades_per_day(self.companies[0])
        num_trades1 = self.models[1].get_trades_per_day(self.companies[0])

        total_trades0 = self.models[0].get_total_trades(self.companies[0], t1=self.start_date, t2=self.end_date)
        total_trades1 = self.models[1].get_total_trades(self.companies[0], t1=self.start_date, t2=self.end_date)

        fx_costs1 = self.models[0].get_fx_costs(self.companies[0], local=True)
        fx_costs2 = self.models[1].get_fx_costs(self.companies[0], local=True)

        interest_costs1 = self.models[0].get_interest_costs(self.companies[0], local=True)
        interest_costs2 = self.models[1].get_interest_costs(self.companies[0], local=True)

        plt.plot(interest_costs2.loc[lastyear, :])
        plt.title('Interest Costs')
        plt.show()

        plt.plot(interest_costs2.loc[lastyear, :].cumsum())
        plt.title('Interest Costs')
        plt.show()

        plt.plot(fx_costs2.loc[lastyear, :])
        plt.title("FX Costs")
        plt.show()

        plt.plot(fx_costs2.loc[lastyear, :].cumsum())
        plt.title("FX Costs")
        plt.show()

        balances0 = balances0.loc[lastyear, :]
        balances1 = balances1.loc[lastyear, :]

        balances0.plot()
        plt.show()

        balances1.plot()
        plt.show()
