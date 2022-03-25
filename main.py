from environment.simulation import Simulation
from evaluation.scoreboard import Scoreboard
from datetime import datetime

if __name__ == '__main__':
    t1 = datetime(year=2017, month=1, day=1)
    t2 = datetime(year=2022, month=1, day=1)

    env = Simulation(start_date=t1, end_date=t2, num_companies=1)
    Scoreboard(env)
