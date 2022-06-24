from environment.simulation import Simulation
from evaluation.scoreboard import Scoreboard
from datetime import datetime
import numpy as np

# np.random.seed(5)

if __name__ == '__main__':
    t1 = datetime(year=2017, month=1, day=1)
    t2 = datetime(year=2022, month=1, day=1)

    start_date = datetime(year=2020, month=1, day=1)
    end_date = datetime(year=2020, month=6, day=1)

    env = Simulation(start_date=start_date, end_date=end_date, t1=t1, t2=t2, num_companies=10)
    scoreboard = Scoreboard(env)

    test_t1 = datetime(year=2020, month=1, day=1)
    test_t2 = datetime(year=2020, month=7, day=1)
    scoreboard.evaluate(t1=test_t1, t2=test_t2)
